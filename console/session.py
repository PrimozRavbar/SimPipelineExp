"""SimSession: one running simulation that a player drives from the console.

No web code here, so it can be tested without a server (see
tests/test_console_session.py). The FastAPI app in console/server.py is a
thin wrapper around one SimSession.

Determinism contract
--------------------
Player actions (retrain, roll back, settings changes...) are queued and
applied only at tick boundaries: after tick T has finished and before tick
T+1 starts. Retraining runs on a background thread while serving continues,
but the new model always deploys at a fixed tick (`deploy_delay` ticks after
the retrain started). If training is not finished in wall time by then, the
simulation clock waits. So the same seed and the same actions at the same
ticks give the same events and the same models, however fast the machine is
and whatever speed the player picked.

Everything that touches the simulation runs under `self.lock`. The
background clock thread releases it between ticks, so API reads see a
consistent state.
"""

import copy
import random

import numpy as np
from pathlib import Path
import statistics
import threading
import time

from ml_system.ann import HNSWIndex
from console import health
from ml_system.build_recommendation_system import build_recommendation_system
from ml_system.faults import FAULT_KINDS, FIXES, FaultInjector
from ml_system.metrics import reference_ctrs
from ml_system.candidate_sources import CoWatchRetrieval, ProfileRetrieval
from ml_system.experiments import Experiment
from ml_system.ranking import (ExplorationReranker, LogisticRanker, MLPRanker, MMRReranker,
                               RerankerChain, WatchedFilter)
from ml_system.model_registry import export_bundle
from ml_system.ann_indexes import INDEX_TYPES, index_embeddings, index_kind, make_index, vectors_per_query
from ml_system.feast import MODES as FEAST_MODES
from ml_system.spark_sim.cluster import TICK_SECONDS, ClusterConfig
from ml_system.spark_sim.engine import ticks_for
from ml_system import training_job, validation
from ml_system.redis import EVICTION_POLICIES
from console.economy import ONE_OFF, PRICES, Economy
from ml_system.online_model import PreferenceModel, PreferenceRanker, PreferenceRetrieval
from ml_system.exploration import POLICIES as EXPLORE_POLICIES, SlotExplorer
from ml_system.cf_model import CFRetrieval, ImplicitMF
from ml_system.training import TwoTowerTrainer, evaluation_loss
from ml_system.two_tower import USER_FEATURE_DEFAULTS, SimpleItemTower, SimpleUserTower, TwoTowerRetrieval
from world.dynamics import DYNAMIC_WORLD
from world.external_data_loader import load_movielens


# Ticks per second at speed 1x (the mockup advanced 1 tick every 500 ms).
BASE_TICKS_PER_SECOND = 2
SPEEDS = (1, 10, 100)
MIN_YIELD_SECONDS = 0.003

# Compute budget: a simple game resource, not a measurement.
BUDGET_TOTAL = 1000
COST_RETRAIN = 40
COST_SPARK = 5
EVAL_SAMPLE = 2000           # training examples model validation scores both models on

DEFAULT_SETTINGS = {
    # Offline schedules (0 = off, the player triggers runs by hand).
    "spark_every": 100,        # run the Spark feature job every N ticks (0 = off)
    "retrain_every": 0,        # start a retrain every N ticks
    "deploy_delay": 50,        # ticks between retrain start and deploy
    # Training
    "epochs": 5,
    "batched_training": True,  # train the towers a batch at a time (fast) or example by example
    # Retrieval training (two-tower); all off by default
    "logq_correction": False,  # correct in-batch negatives for item popularity (sampling bias)
    "random_negatives": 0,     # extra random catalog negatives per example
    "graded_positives": False, # weight clicks by watch completion and rating
    "tower_history": False,    # user tower also sees the recent-watch history
    "train_from_scratch": False,  # new towers every retrain instead of continuing the last ones
    "tower_engagement": False, # user tower sees per-genre click-through (skips) and profile strength
    "tower_demographics": False,  # user tower sees age bracket, gender, occupation
    "tower_shared_ids": False, # movie IDs in the movie tower, shared with the user's watch history
    "negatives": 5,            # sampled negatives per positive
    # Training job on a simulated GPU cluster (ml_system/training_job.py) and
    # the checks of the retrain workflow (ml_system/validation.py).
    "train_gpus": 1,
    "train_gpu_memory_gb": 16,
    "train_batch_size": 32,    # examples per training step (real: changes the model)
    "learning_rate": "0.001",  # Adam learning rate (real: too high and training diverges)
    "train_checkpoint_minutes": 10,  # a preempted job resumes from its last checkpoint (0 = none)
    "validate_data": True,     # check the training set against the last good one before training
    "validate_model": True,    # check the trained model (loss, embeddings) before registering it
    # Serving
    "slate_size": 10,
    "num_candidates": 100,
    "rebuild_index": True,     # rebuild the ANN index with the new item tower on deploy
    "ranker": False,           # train and serve a ranking stage after retrieval
    "ranker_mlp": True,        # ranker model: MLP with cross/recent-history features (off: logistic)
    "ranker_genre_features": True,  # MLP also sees the taste match genre by genre (feature set v3)
    "ranker_multitask": False, # MLP also predicts watch completion; orders by p(click) x completion
    "ranker_signals": False,   # MLP also sees movie signals from the logs: click rate, completion, rating, co-watching (v4)
    "ranker_tower_score": False,  # MLP also sees the two-tower retrieval score (v5; needs ranker_signals)
    "cowatch_candidates": 0,   # extra candidates co-watched with the user's recent watches (0 = off)
    # Preference model (ml_system/online_model.py): learns from every logged
    # signal, online between retrains and refitted at every retrain; ranks
    # the candidates and adds its own best movies from the whole catalog.
    "preference_model": False,
    "preference_candidates": 50,
    "preference_factors": 0,       # hidden taste factors per user and movie (0 = genres and quality only)
    "preference_history": False,   # it also learns from times shown / already watched (repeat fatigue)
    "preference_half_life": 0,
    "preference_optimism_pct": 0,
    "preference_taste_half_life": 0,
    "cf_model": False,         # collaborative filtering (implicit ALS) refitted at every retrain; a preference-model input
    "cf_candidates": 0,        # extra candidates from collaborative filtering (needs cf_model; 0 = off)
    "cf_factors": 16,          # hidden taste numbers per user and movie
    "preference_browse_memory_pct": 80,  # how much of the browse history is kept per new browse (80 = about the last 5)
    "preference_grade_weight_pct": 100,  # weight of watch completion and ratings vs clicks in training
    "preference_genre_optimism_pct": 0,  # per-user bonus for genres the user has rarely been shown, x0.01 logits  # user tastes follow recent behaviour: rows halve every N ticks (0 = off)  # weight of a skip per earlier impression of the movie (100 = all skips equal)  # exploration bonus for rarely shown movies (UCB-style), x0.01 logits
    "ranker_preference_score": False,  # MLP ranker also sees the preference model's score (v6; logged at serving)     # refits weight a row half as much every N ticks of age (0 = all equal)
    # Personalisation
    "profile_candidates": 0,   # extra candidates from the user's genre profile (0 = two-tower only)
    "rich_profiles": False,    # profiles from clicks, watch completion and ratings, recent first
    "onboarding": False,       # new users pick favourite genres before their first slate
    "filter_watched": False,   # drop movies the user already watched (re-ranker)
    "diversity_pct": 0,        # MMR on genres: 0 = pure relevance, 100 = most varied slates
    "exploration_pct": 0,      # share of slate slots given to random lower-ranked candidates
    # Targeted exploration (ml_system/exploration.py; needs the preference model)
    "explore_slots": 0,        # slots per slate an exploration pick may take (0 = off)
    "explore_rate_pct": 10,    # share of slates that explore
    "explore_boost_pct": 50,   # ... for users the drift detector flags
    "explore_policy": "thompson",  # or "uniform"
    "explore_ips": True,       # train on explored picks with inverse propensity weights
    # Canary: a new model first serves canary_share % of users for
    # canary_ticks ticks, then is promoted unless it is significantly worse.
    "canary_share": 0,
    "canary_ticks": 150,
    # Offline evaluation gate: hold out the newest holdout_pct % of events,
    # score Recall@20 of the new and the production model on their clicks,
    # and refuse to deploy a model more than gate_tolerance_pct % worse.
    "eval_gate": False,
    "holdout_pct": 10,
    "gate_tolerance_pct": 10,
    # Alert rule: open an incident when rolling CTR drops more than
    # alert_drop_pct % within alert_ticks ticks after a deploy.
    "alert_drop_pct": 20,
    "alert_ticks": 100,
    "alert_window": 50,
    # Operations
    "flink_capacity": 0,       # Flink events processed per tick (0 = unlimited)
    "lag_alert": 150,          # incident when Flink's Kafka lag exceeds this
    "skew_alert_pct": 60,      # incident when online/offline feature skew exceeds this %
    "chaos_every": 0,          # a hidden random fault about every N ticks (0 = off)
    "spark_throughput": 0,     # legacy: events a Spark job processes per tick (0 = use the cluster model)
    # Spark cluster (ml_system/spark_sim): a job's duration and failures come
    # from its tasks on this cluster; 1 tick ~ 15 minutes, and a job that
    # finishes inside its first tick is ready at once.
    "spark_executors": 4,
    "spark_executor_cores": 4,
    "spark_executor_memory_gb": 8,
    "spark_shuffle_partitions": 200,
    "spark_aqe": True,         # adaptive query execution: coalesce small partitions, split skewed ones
    "spark_speculation": True, # re-launch straggling tasks on another executor
    "spark_task_max_failures": 4,  # attempts per task before the job is aborted
    "spark_incremental": True, # process only the events since the last run (a full backfill when needed)
    "spark_wait_for_data": True,   # wait until the lake has the latest data before running (sensor)
    "spark_sensor_timeout": 8, # ... and fail the job after waiting this many ticks
    "lake_compact_every": 100, # merge the data lake's small files every N ticks (0 = never)
    "freshness_alert": 400,    # incident when offline features are older than this (ticks)
    # Infrastructure (defaults = the original behaviour: nothing limited)
    "kafka_partitions": 3,
    "kafka_replication": 1,    # copies of each partition (a broker outage stalls unreplicated ones)
    "kafka_brokers": 3,
    "kafka_retention_ticks": 0,   # delete messages older than this (0 = keep forever)
    "flink_parallelism": 1,    # task slots (changing it = savepoint + restart, 2 ticks)
    "flink_slot_capacity": 0,  # events one slot processes per tick (0 = unlimited)
    "redis_max_keys": 0,       # memory limit in keys (0 = unlimited)
    "redis_eviction": "allkeys-lru",
    "redis_ttl_ticks": 0,      # user features expire this long after their last update (0 = never)
    "feast_mode": "direct",    # "materialized": serve batch features from the online copy
    "feast_materialize_every": 100,
    "kserve_min_replicas": 1,
    "kserve_max_replicas": 1,
    "kserve_replica_capacity": 0,  # requests per replica per tick (0 = unlimited)
    "kserve_target_pct": 70,   # autoscaler target utilisation
    "kserve_cold_start_ticks": 3,
    "ann_nprobe": 4,           # IVF index: cells searched per query
    # World (simulated users)
    "taste_drift_every": 0,    # every N ticks some users' taste drifts (0 = off)
    "taste_drift_share_pct": 5,
    "taste_drift_mode": "abrupt",  # "gradual": small steps towards a related genre (the dynamic world's default)
    "browse_pct": 30,          # share of visits where the user first browses a genre (BrowseEvent)
    "rating_pct": 10,          # share of watched movies the user rates
    "interest_save_pct": 100,  # an examined, unclicked relevant movie gets saved (x its relevance lift)
    "interest_dismiss_pct": 50, # an examined irrelevant movie gets "not interested"
    "session_slates": 1,       # slates per visit (the next row; later slates skip movies already shown)
    "fatigue_pct": 0,          # click probability lost per repeat impression
    "fatigue_recovery_ticks": 0,  # fatigue halves every N ticks since the last impression (0 = never)
    "rewatch_pct": 100,        # click probability kept for an already-watched movie
    "users_joining_per_100_ticks": 0,
    "viral_every": 0,          # every N ticks a movie goes viral: everyone talks about it (0 = off)
    "viral_duration": 300,     # ... for this many ticks
}

# Presets. "plain": every model feature off (the original console). "best":
# the setup with the best measured CTR (see CLAUDE.md, "Default setup");
# the default. Players can change any setting afterwards.
PLAIN_SETTINGS = dict(DEFAULT_SETTINGS)
BEST_SETTINGS = {
    **PLAIN_SETTINGS,
    "ranker": True, "ranker_mlp": True, "ranker_genre_features": True,
    "ranker_signals": True, "ranker_tower_score": True,
    "profile_candidates": 30, "cowatch_candidates": 30,
    "rich_profiles": True, "onboarding": True, "logq_correction": True,
    "diversity_pct": 15,
    "preference_model": True, "preference_history": True, "ranker_preference_score": True,
}
PRESETS = {"plain": PLAIN_SETTINGS, "best": BEST_SETTINGS}

# The best setup in the dynamic world adds targeted exploration (drifting
# users; measured in CLAUDE.md "Exploration"). In the static world, where
# tastes never move, exploration only costs clicks, so it stays off.
BEST_DYNAMIC_EXTRAS = {"explore_slots": 1, "explore_rate_pct": 10, "explore_boost_pct": 50,
                       "explore_policy": "thompson", "explore_ips": True}
DEFAULT_SETTINGS = dict(BEST_SETTINGS)

# (min, max) for numeric settings; bools are listed in BOOL_SETTINGS.
SETTING_LIMITS = {
    "spark_every": (0, 100000),
    "retrain_every": (0, 100000),
    "deploy_delay": (1, 10000),
    "epochs": (1, 50),
    "negatives": (1, 9),
    "train_gpus": (1, 64),
    "train_gpu_memory_gb": (4, 192),
    "train_batch_size": (1, 8192),
    "train_checkpoint_minutes": (0, 1440),
    "slate_size": (1, 100),
    "num_candidates": (1, 500),
    "alert_drop_pct": (1, 100),
    "alert_ticks": (10, 10000),
    "alert_window": (5, 1000),
    "profile_candidates": (0, 100),
    "random_negatives": (0, 20),
    "cowatch_candidates": (0, 100),
    "preference_candidates": (0, 500),
    "preference_factors": (0, 64),
    "preference_half_life": (0, 100000),
    "preference_optimism_pct": (0, 500),
    "preference_taste_half_life": (0, 100000),
    "preference_browse_memory_pct": (0, 99),
    "cf_candidates": (0, 200),
    "cf_factors": (2, 64),
    "preference_grade_weight_pct": (0, 1000),
    "preference_genre_optimism_pct": (0, 500),
    "diversity_pct": (0, 100),
    "exploration_pct": (0, 50),
    "canary_share": (0, 100),
    "holdout_pct": (1, 50),
    "gate_tolerance_pct": (0, 100),
    "canary_ticks": (10, 10000),
    "flink_capacity": (0, 100000),
    "lag_alert": (10, 1000000),
    "skew_alert_pct": (1, 100),
    "chaos_every": (0, 100000),
    "spark_throughput": (0, 10000000),
    "spark_executors": (1, 64),
    "spark_executor_cores": (1, 32),
    "spark_executor_memory_gb": (1, 256),
    "spark_shuffle_partitions": (1, 5000),
    "spark_task_max_failures": (1, 20),
    "spark_sensor_timeout": (1, 10000),
    "lake_compact_every": (0, 100000),
    "viral_every": (0, 100000),
    "viral_duration": (10, 100000),
    "freshness_alert": (10, 100000),
    "taste_drift_every": (0, 100000),
    "taste_drift_share_pct": (1, 100),
    "browse_pct": (0, 100),
    "rating_pct": (0, 100),
    "interest_save_pct": (0, 100),
    "interest_dismiss_pct": (0, 100),
    "session_slates": (1, 5),
    "explore_slots": (0, 5),
    "explore_rate_pct": (0, 100),
    "explore_boost_pct": (0, 100),
    "fatigue_pct": (0, 90),
    "fatigue_recovery_ticks": (0, 100000),
    "kafka_partitions": (1, 64),
    "kafka_replication": (1, 5),
    "kafka_brokers": (1, 9),
    "kafka_retention_ticks": (0, 100000),
    "flink_parallelism": (1, 64),
    "flink_slot_capacity": (0, 1000000),
    "redis_max_keys": (0, 10000000),
    "redis_ttl_ticks": (0, 100000),
    "feast_materialize_every": (0, 100000),
    "kserve_min_replicas": (0, 50),
    "kserve_max_replicas": (1, 50),
    "kserve_replica_capacity": (0, 100000),
    "kserve_target_pct": (10, 100),
    "kserve_cold_start_ticks": (0, 1000),
    "ann_nprobe": (1, 256),
    "rewatch_pct": (0, 100),
    "users_joining_per_100_ticks": (0, 10000),
}
BOOL_SETTINGS = {"rebuild_index", "ranker", "ranker_mlp", "filter_watched", "eval_gate", "batched_training",
                 "ranker_genre_features", "rich_profiles", "onboarding",
                 "logq_correction", "graded_positives", "tower_history", "train_from_scratch",
                 "ranker_multitask", "ranker_signals", "tower_engagement", "tower_demographics",
                 "tower_shared_ids", "ranker_tower_score", "preference_model",
                 "preference_history", "ranker_preference_score", "explore_ips", "cf_model",
                 "spark_aqe", "spark_speculation", "spark_incremental", "spark_wait_for_data",
                 "validate_data", "validate_model"}

# Settings that take one of a few strings.
LEARNING_RATES = ("0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0")
CHOICE_SETTINGS = {"redis_eviction": EVICTION_POLICIES, "feast_mode": FEAST_MODES,
                   "learning_rate": LEARNING_RATES,
                   "taste_drift_mode": ("abrupt", "gradual"), "explore_policy": EXPLORE_POLICIES}

# User features that expire with redis_ttl_ticks.
TTL_FEATURES = ("genre_preferences", "activity_level", "interactions", "average_rating",
                "recent_watches", "recent_slates", "genre_engagement", "profile_strength")

WORLDS = ("static", "dynamic")

# Settings that the dynamic world preset starts from.
DYNAMIC_SETTINGS = {
    "taste_drift_every": DYNAMIC_WORLD["taste_drift_every"],
    "taste_drift_share_pct": round(DYNAMIC_WORLD["taste_drift_share"] * 100),
    "taste_drift_mode": DYNAMIC_WORLD["taste_drift_mode"],
    "fatigue_pct": round(DYNAMIC_WORLD["fatigue"] * 100),
    "fatigue_recovery_ticks": DYNAMIC_WORLD["fatigue_recovery"],
    "rewatch_pct": round(DYNAMIC_WORLD["rewatch"] * 100),
    "users_joining_per_100_ticks": round(DYNAMIC_WORLD["users_joining_per_tick"] * 100),
}

# Chaos: how long each kind of random fault lasts (None = until fixed).
CHAOS_DURATIONS = {
    "flink_outage": 150,
    "feature_corruption": None,
    "index_id_mismatch": None,
    "model_server_outage": 120,
    "traffic_spike": 80,
    "spark_failure": 300,
    "spark_executor_loss": 200,
    "spark_slow_node": None,
    "spark_cluster_busy": 150,
    "lake_ingestion_lag": None,
    "viral_movie": 300,
    "gpu_preemption": 300,
    "feature_nulls": 200,
    "scheduler_down": None,
}

RUNBOOKS = {
    "budget": [
        "Open Pipeline > Costs: the bill per component per tick is listed.",
        "Cut what you do not need: KServe replicas, Flink slots, Kafka brokers, or swap to a cheaper ANN index.",
        "Clicks earn money: a better model pays for its infrastructure.",
    ],
    "kafka_partitions": [
        "A broker is down and some partitions have no other replica.",
        "Restart the broker (Repair: restart Kafka broker), or wait for it to come back.",
        "Prevent it: replication factor 2 or more keeps partitions online when one broker fails.",
    ],
    "serving_capacity": [
        "More requests than KServe replicas can take: the rest get the popularity fallback.",
        "Raise max replicas (the autoscaler adds them after a cold start) or the capacity per replica.",
        "Traffic spikes need headroom: a lower target utilisation scales up earlier.",
    ],
    "ctr_drop": [
        "Check whether the drop lines up with a deploy (Monitoring, CTR chart). If it does, roll back first.",
        "Otherwise check the health tiles on Monitoring: Flink lag, fallback share, index self-test, feature skew.",
        "Fix the component that is unhealthy (Pipeline: Restart Flink, Rebuild index, Restart model server).",
        "If everything is healthy, look for drift: the world may have changed since the model was trained. Retrain.",
    ],
    "flink_lag": [
        "Open Pipeline, Stream features (Flink): is it processing anything?",
        "If it processes nothing, restart Flink. If it is up but behind, raise its capacity.",
        "While Flink is behind, online features are stale: expect CTR to sag for active users.",
    ],
    "fallback": [
        "The model server cannot reach two-tower retrieval and serves its popularity fallback.",
        "Restart the model server (Pipeline, Model server).",
    ],
    "index_health": [
        "The ANN index self-test fails: golden queries recorded when the index was built now return other movies.",
        "The index's id mapping is broken. Rebuild the index (Pipeline, ANN index) or deploy a model with a fresh index.",
    ],
    "feature_skew": [
        "Online user features (Redis, used by serving) disagree with offline ones (feature store).",
        "A bad feature-job deploy is the usual cause: restart Flink (Pipeline, Stream features) to redeploy it.",
        "Then run the Spark feature job so both sides are recomputed.",
    ],
    "canary": [
        "The canary served the new model to a share of users and lost to the current model with 95% confidence.",
        "It was rolled back automatically: only the canary users were affected.",
        "Compare the two versions on Models (training data, epochs, ranker) before trying again.",
    ],
    "freshness": [
        "Offline features have not been recomputed for a long time: training and serving fallbacks use old data.",
        "Check the Spark job on Pipeline (Batch features): is it scheduled, and do its runs fail?",
        "If runs fail, wait for the cluster to recover, then run the job by hand.",
    ],
    "spark_job": [
        "Open Pipeline, Batch features (Spark): the last job's stages, tasks and log show where it failed.",
        "Out of memory: a task held more rows than its executor memory allows. Raise executor memory, use "
        "fewer cores per executor, more shuffle partitions, or switch AQE on (it splits skewed partitions).",
        "Executors lost: spot nodes are being reclaimed. Tasks are retried and lost shuffle output recomputed "
        "(slower jobs); the job gives up after max(2 x executors, 3) lost executors. More executors and "
        "incremental (short) runs ride it out.",
        "Sensor timed out: the data lake is behind (see the lake's Kafka lag). Restart the lake sink.",
        "The feature store keeps the last good version: a failed job never publishes half its output.",
    ],
    "spark_slow": [
        "The last Spark job took more than a tick: features (and a retrain waiting for them) land later.",
        "Check the stage table: one slow task (skew ratio) means a hot key or a slow node; all tasks slow "
        "means too few executors, spilling (memory) or a full recompute.",
        "Fixes: more executors or memory, AQE and speculation on, incremental runs, replace a degraded node.",
    ],
    "workflow": [
        "Open Pipeline, Training: the last retrain workflow shows which step failed and why.",
        "Data validation failed: the training set looks broken (too few examples, labels missing, many "
        "empty profiles). Find the cause first: an empty-profile jump is a broken feature job; fix it with "
        "Backfill features (Pipeline, Batch features) and retrain.",
        "GPU out of memory: lower the batch size or use GPUs with more memory.",
        "Preempted too often: keep checkpoints on (a preempted job resumes from them) or wait.",
        "Model validation failed: the loss is not a number or went up, or the new model is worse than the "
        "serving one on the same new data (a learning rate too high or too low trains a worse model). "
        "Fix the setting and retrain.",
        "The serving model is unchanged: a failed workflow never deploys.",
    ],
    "schedule_missed": [
        "Scheduled runs (Spark, retrains) did not start: the workflow scheduler is down, and nothing "
        "else tells you. Features and models silently go stale.",
        "Restart the scheduler (Pipeline, Training), then run what was missed by hand.",
    ],
    "lake_ingestion": [
        "The data lake sink is not writing: events pile up in Kafka (its lag) and batch jobs see old data.",
        "Restart the lake sink (Pipeline, Data lake). With 'wait for data' on, Spark jobs wait for it; off, "
        "they run on incomplete data and succeed silently.",
    ],
    "drift": [
        "What users click has shifted away from what the model was trained on.",
        "Retrain on recent data (Models, Retrain now) and compare online CTR before and after.",
    ],
}


RUNBOOK_CTR_DROP = [
    "Check whether the drop lines up with a deploy (Monitoring, CTR chart).",
    "If it does, roll back first and investigate after (Models, Roll back).",
    "Compare the model and index versions the server uses (Pipeline: Model server and ANN index).",
    "Check how much data the new model was trained on and whether features are fresh (Models, Pipeline: Spark).",
]


class ActionError(ValueError):
    """An action the session cannot apply (bad input, or not allowed now)."""


class ModelVersion:
    """One deployable model: user tower + item tower + ANN index."""

    def __init__(self, version, user_tower, item_tower, index, ranker=None, preferences=None, cf=None, **info):
        self.version = version
        self.ranker = ranker
        self.preferences = preferences        # fitted PreferenceModel (or None)
        self.cf = cf                          # fitted ImplicitMF (or None)
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.index = index
        self.info = info

    def summary(self):
        return {"version": self.version, **self.info}


class _ItemFeatureSnapshot:
    """Item features frozen at retrain start, so the background thread never
    reads the live feature store (a Spark run could change it meanwhile)."""

    def __init__(self, feature_store):
        self.item_features = {
            movie_id: copy.deepcopy(feature_store.get_item_features(movie_id))
            for movie_id in feature_store.item_features
        }


class SimSession:
    """Owns one simulation and runs its clock on a background thread.

    `data` is the output of load_movielens(); it is loaded fresh by default
    because the World mutates user and movie state as it runs.
    """

    def __init__(
        self,
        seed=42,
        users_per_tick=20,
        settings=None,
        data=None,
        log_metrics=True,
        runs_dir="runs",
        run_id=None,
        training_wait_timeout=None,
        world="static",
        taste="genre",
        preset="best",
    ):
        if world not in WORLDS:
            raise ActionError("world must be one of %s" % (WORLDS,))
        if taste not in ("genre", "rich", "personal"):
            raise ActionError('taste must be "genre", "rich" or "personal"')
        self.seed = seed
        self.world_kind = world
        self.taste_kind = taste
        self.users_per_tick = users_per_tick
        if preset not in PRESETS:
            raise ActionError("preset must be one of %s" % sorted(PRESETS))
        self.preset = preset
        self.settings = dict(PRESETS[preset])
        if world == "dynamic":
            self.settings.update(DYNAMIC_SETTINGS)
            if preset == "best":
                self.settings.update(BEST_DYNAMIC_EXTRAS)
        if settings:
            self._check_settings(settings)
            self.settings.update(settings)

        data = data if data is not None else load_movielens()

        self.system = build_recommendation_system(
            movie_objects=data["movie_objects"],
            user_objects=data["user_objects"],
            user_states=data["user_states"],
            movie_states=data["movie_states"],
            seed=seed,
            num_candidates=self.settings["num_candidates"],
            slate_size=self.settings["slate_size"],
            users_per_tick=users_per_tick,
            log_metrics=log_metrics,
            runs_dir=runs_dir,
            run_id=run_id,
            # Always register WorldDynamics so world settings are live; a
            # static world gets a neutral one (no RNG use outside its own).
            world_dynamics=(dict(DYNAMIC_WORLD) if world == "dynamic"
                            else {"taste_drift_every": 0}),
            taste=taste,
        )

        self.sim = self.system["sim"]
        self.dynamics = self.system["world_dynamics"]
        self.simulator = self.system["world"].user_simulator
        self.flink = self.sim.get_service("Flink")
        self.flink.track_watched = True      # for the watched-movies filter
        self.kafka = self.sim.get_service("Kafka")

        # Fault injection, registered last (after the metrics logger).
        self.faults = FaultInjector(self.system)
        self.sim.register(self.faults)
        self.chaos_rng = random.Random("chaos-%d" % seed)
        self.next_chaos_tick = None
        self.drills = set()
        # Exploration draws from its own seeded RNG (never the shared stream).
        self.explore_rng = random.Random("explore-%d" % seed)
        self.explorer = None

        self.logger = self.system["metrics_logger"]
        self.tracker = self.system["ctr_tracker"]
        self.retriever = self.system["retriever"]
        self.model_server = self.system["model_server"]
        self.offline = self.system["offline_pipeline"]

        if self.logger is not None:
            self.reference = dict(self.logger.run_info["reference"])
        else:
            self.reference = reference_ctrs(
                self.system["world"], slate_size=self.settings["slate_size"], seed=seed
            )

        self.lock = threading.RLock()
        self.training_wait_timeout = training_wait_timeout

        # Clock thread state
        self.playing = False
        self.speed = 1
        self._stop = threading.Event()
        self._thread = None
        self.waiting_for_training = False
        self.clock_error = None

        # Actions queued for the next tick boundary.
        self._pending = []
        self._next_action_id = 1

        # Models. Version 0 is the untrained bootstrap model.
        bootstrap = ModelVersion(
            0,
            self.retriever.user_tower,
            self.retriever.item_tower,
            self.retriever.ann_index,
            started_tick=0,
            trained_tick=0,
            deployed_tick=0,
            training_examples=0,
            epochs=None,
            negatives=None,
            train_seconds=None,
            recall_at_20=None,
            note="bootstrap model (untrained, explores)",
        )
        self.versions = {0: bootstrap}
        self.serving_version = 0
        self.index_version = 0
        self.deploy_history = [(0, 0)]   # (tick, version)
        self.retrain = None              # the in-flight retrain, if any
        self.canary = None               # {"experiment", "rebuild_index", "by"} while a canary runs
        self.tick_versions = {}          # tick -> model version that served it

        # Offline jobs
        self.last_spark_tick = None
        self.spark_job = None
        self.spark_runs = 0
        self.spark_failures = 0
        self.last_spark_error = None     # {"tick", "reason", "error", "by"} of the last failed job
        self._spark_bill = 0.0           # money for Spark jobs finished since the last tick was billed
        self._train_bill = 0.0           # money for GPU time of training jobs, billed like Spark
        self.workflow_runs = []          # finished retrain workflow runs, newest last
        self.last_workflow_failure = None
        self.data_reference = None       # stats of the last training set that passed data validation
        self.system["training_conditions"] = training_job.TrainingConditions()
        self.system["scheduler"] = {"up": True}
        self.missed_runs = []            # (tick, "spark" | "retrain") skipped while the scheduler was down

        # Budget, event log, incidents
        self.budget_used = 0
        self.log = []
        self.incidents = []
        self._watch = None               # deploy being watched by the alert rule

        # Serving latency: wall-clock ms of ModelServer.recommend per request.
        self.latency = {}                # tick -> list of ms
        self._wrap_recommend()

        # Monitoring: latest health readings, open incident per rule.
        self.health = {"index_self_test": None, "feature_skew": None, "flink_lag": 0,
                       "fallback_share": 0.0, "coverage": None, "drift_psi": None, "feature_age": 0}
        self.fallback_ticks = {}         # tick -> (requests, fallback requests)
        self.open_rules = {}             # rule -> incident
        self.golden = {}                 # index version -> golden queries
        # Feature log: the retrieval score of every served (tick, user,
        # movie), computed by the model that served it (for the v5 ranker).
        self.score_log = {}
        self.preference_log = {}         # (tick, user, movie) -> preference model score at serving
        self.model_server.score_logger = self._log_scores
        self.drift_reference = None      # genre mix of clicks just after the last deploy
        # Infrastructure and money.
        self.economy = Economy()
        self.index_kind = "hnsw"         # ANN index type new models are built with
        self.feast = self.system["feast"]
        self.kserve = self.system["kserve"]
        self._infra_applied = False
        self.cf_live = None              # deployed collaborative-filtering model
        self._materializations_billed = 0
        self._apply_live_settings()

        self._log("system", "Simulation built: seed %d, %d users per tick." % (seed, users_per_tick))
        self._log("deploy", "Model v0 (bootstrap, untrained) serving.")

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------

    @property
    def tick(self):
        return self.sim.clock.now

    def start(self):
        """Start the background clock thread (paused until play())."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="sim-clock", daemon=True)
            self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.retrain is not None and self.retrain["thread"] is not None:
            self.retrain["thread"].join()

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def set_speed(self, speed):
        speed = int(speed)
        if speed not in SPEEDS:
            raise ActionError("speed must be one of %s" % (SPEEDS,))
        self.speed = speed

    def step(self, n=1):
        """Advance n ticks synchronously (tests, scripted runs). Waits for
        training where a deploy is due."""
        for _ in range(int(n)):
            self._wait_for_training()
            with self.lock:
                self._step_locked()
        return self.tick

    def _run(self):
        next_time = time.monotonic()
        while not self._stop.is_set():
            if not self.playing:
                next_time = time.monotonic()
                self._stop.wait(0.05)
                continue

            if self._training_blocks_next_tick():
                # Deterministic deploy: the clock waits for training.
                self.waiting_for_training = True
                self._stop.wait(0.05)
                next_time = time.monotonic()
                continue
            self.waiting_for_training = False

            try:
                with self.lock:
                    self._step_locked()
            except Exception as error:  # keep the server alive; show the error
                self.playing = False
                self.clock_error = repr(error)
                with self.lock:
                    self._log("system", "Clock stopped by an error: %r" % (error,), cls="s-crit")
                continue

            next_time += 1.0 / (BASE_TICKS_PER_SECOND * self.speed)
            delay = next_time - time.monotonic()
            if delay <= 0:
                # Running behind (ticks are slower than the speed asks):
                # don't build up a backlog.
                next_time = time.monotonic()
            # Always let go for a moment: locks are not fair, and without a
            # pause API requests waiting for the lock would starve.
            self._stop.wait(max(delay, MIN_YIELD_SECONDS))

    def _training_blocks_next_tick(self):
        r = self.retrain
        return (
            r is not None
            and r["thread"] is not None
            and self.tick >= r["deploy_tick"]
            and not r["done"].is_set()
        )

    def _wait_for_training(self):
        if self._training_blocks_next_tick():
            self.waiting_for_training = True
            if not self.retrain["done"].wait(self.training_wait_timeout):
                raise TimeoutError("training did not finish in time")
            self.waiting_for_training = False

    def _step_locked(self):
        """Apply the tick boundary (scheduled jobs, deploys, player actions),
        then run one tick."""
        self._boundary()
        self.tick_versions[self.tick + 1] = self.serving_version
        before = (self.model_server.requests, self.model_server.fallback_requests)
        self.sim.tick()
        self.fallback_ticks[self.tick] = (self.model_server.requests - before[0],
                                          self.model_server.fallback_requests - before[1])
        self._account()
        self._check_alert()
        if self.canary is not None:
            self._canary_tick()
        self._monitor()

    def _boundary(self):
        now = self.tick

        # 1. A due deploy goes first, so actions at this boundary see it.
        # 0. A Spark job waiting for its input starts once the lake has
        # caught up; a job that is due finishes (a retrain may be waiting).
        if self.spark_job is not None and self.spark_job["status"] == "waiting":
            self._check_spark_sensor()
        if self.spark_job is not None and self.spark_job["status"] == "running" \
                and now >= self.spark_job["done_tick"]:
            self._finish_spark()
        s = self.settings
        if now > 0 and s["lake_compact_every"] and now % s["lake_compact_every"] == 0:
            before, after = self.system["datalake"].compact(now)
            if before != after:
                self._log("job", "Data lake compaction: %d files -> %d." % (before, after))

        if self.retrain is not None and self.retrain["fail_at"] is not None \
                and self.retrain["thread"] is None and now >= self.retrain["fail_at"]:
            self._training_failed(self.retrain)
        if self.retrain is not None and self.retrain["thread"] is not None \
                and now >= self.retrain["deploy_tick"]:
            self._finish_retrain()

        # 2. Player actions, in the order they were submitted.
        pending, self._pending = self._pending, []
        for action in pending:
            try:
                message = self._apply(action["name"], action["params"])
                action.update(status="applied", tick=now, message=message)
            except ActionError as error:
                action.update(status="rejected", tick=now, message=str(error))
                self._log("action", "Rejected %s: %s" % (action["name"], error), cls="s-warn")

        # 3. Schedules from the settings.
        s = self.settings
        self._chaos(now)
        scheduler_up = self.system["scheduler"]["up"]
        if now > 0 and s["spark_every"] and now % s["spark_every"] == 0 and not scheduler_up:
            self.missed_runs.append((now, "spark"))          # silently: nothing runs, nothing is logged
        elif now > 0 and s["spark_every"] and now % s["spark_every"] == 0:
            self.missed_runs = []                            # the schedule runs again
            try:
                self._run_spark(by="schedule")
            except ActionError as error:
                self._log("job", "Scheduled Spark run skipped: %s" % error, cls="s-warn")
        if now > 0 and s["retrain_every"] and now % s["retrain_every"] == 0 and not scheduler_up:
            self.missed_runs.append((now, "retrain"))
        elif now > 0 and s["retrain_every"] and now % s["retrain_every"] == 0:
            self.missed_runs = []
            if self.retrain is not None:
                self._log("train", "Scheduled retrain skipped: a training run is still in progress.", cls="s-warn")
            else:
                try:
                    self._start_retrain(by="schedule")
                except ActionError as error:
                    self._log("train", "Scheduled retrain skipped: %s" % error, cls="s-warn")

    # ------------------------------------------------------------------
    # Player actions
    # ------------------------------------------------------------------

    ACTIONS = ("retrain", "rollback", "spark", "settings", "fault",
               "promote_canary", "abort_canary", "export", "swap") + tuple(FIXES)

    # Components that can be swapped for another that does the same job.
    SWAPPABLE = {"ann_index": INDEX_TYPES}

    def submit(self, name, **params):
        """Queue an action for the next tick boundary. Returns its record;
        `status` becomes "applied" or "rejected" once the boundary runs."""
        if name not in self.ACTIONS:
            raise ActionError("unknown action %r" % name)
        if name == "settings":
            self._check_settings(params)
        if name == "fault" and params.get("kind") not in FAULT_KINDS:
            raise ActionError("fault kind must be one of %s" % sorted(FAULT_KINDS))
        if name == "swap":
            options = self.SWAPPABLE.get(params.get("component"))
            if options is None:
                raise ActionError("component must be one of %s" % sorted(self.SWAPPABLE))
            if params.get("to") not in options:
                raise ActionError("%s must be one of %s" % (params.get("component"), sorted(options)))
        with self.lock:
            action = {
                "id": self._next_action_id,
                "name": name,
                "params": params,
                "status": "pending",
                "submitted_tick": self.tick,
            }
            self._next_action_id += 1
            self._pending.append(action)
            return action

    def _apply(self, name, params):
        if name == "retrain":
            return self._start_retrain(by="you")
        if name == "rollback":
            return self._rollback()
        if name == "spark":
            return self._run_spark(by="you")
        if name == "settings":
            return self._apply_settings(params)
        if name == "fault":
            return self._drill(params["kind"], params.get("duration"))
        if name in FIXES:
            return self._fix(name)
        if name == "export":
            return self._export(params.get("version", self.serving_version))
        if name == "promote_canary":
            return self._end_canary(promote=True, by="you")
        if name == "abort_canary":
            return self._end_canary(promote=False, by="you")
        if name == "swap":
            return self._swap(params["component"], params["to"])
        raise ActionError("unknown action %r" % name)

    # Settings ----------------------------------------------------------

    def _check_settings(self, values):
        for key, value in values.items():
            if key not in DEFAULT_SETTINGS:
                raise ActionError("unknown setting %r" % key)
            if key in BOOL_SETTINGS:
                if not isinstance(value, bool):
                    raise ActionError("%s must be true or false" % key)
                continue
            if key in CHOICE_SETTINGS:
                if value not in CHOICE_SETTINGS[key]:
                    raise ActionError("%s must be one of %s" % (key, ", ".join(CHOICE_SETTINGS[key])))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise ActionError("%s must be a whole number" % key)
            low, high = SETTING_LIMITS[key]
            if not low <= value <= high:
                raise ActionError("%s must be between %d and %d" % (key, low, high))

        slate = values.get("slate_size", self.settings["slate_size"] if hasattr(self, "settings") else 10)
        cands = values.get("num_candidates", self.settings["num_candidates"] if hasattr(self, "settings") else 100)
        if slate > cands:
            raise ActionError("slate size cannot exceed retrieved candidates")
        current = self.settings if hasattr(self, "settings") else DEFAULT_SETTINGS
        merged = {**current, **values}
        if merged["kafka_replication"] > merged["kafka_brokers"]:
            raise ActionError("Kafka replication cannot exceed the number of brokers")
        if merged["kserve_min_replicas"] > merged["kserve_max_replicas"]:
            raise ActionError("KServe min replicas cannot exceed max replicas")

    def _apply_settings(self, values):
        self._check_settings(values)
        changed = {k: v for k, v in values.items() if self.settings[k] != v}
        for key, value in changed.items():
            self.settings[key] = value if key in BOOL_SETTINGS or key in CHOICE_SETTINGS else int(value)
        self._apply_live_settings()
        if changed:
            text = ", ".join("%s=%s" % (k, v) for k, v in sorted(changed.items()))
            self._log("config", "Settings changed: %s." % text, cls="s-accent")
            if self.logger is not None:
                self.logger.log_event("settings", self.tick, **changed)
        return "settings applied"

    # Spark -------------------------------------------------------------

    def _run_spark(self, by, events=None, charge=True, then=None):
        """Submit the offline feature job (Spark) over the data lake.

        The job first waits for complete input (spark_wait_for_data: the
        lake must have every tick so far; it fails after
        spark_sensor_timeout ticks). Then it is planned and simulated on
        the cluster (ml_system/spark.py): its duration in ticks is the
        simulated time / TICK_SECONDS, rounded down, so a job that
        finishes inside its first tick is ready at once. Its features land
        when it finishes. `then(ok, events)` runs at completion (a retrain
        uses it to start training). Every retrain runs this as its first
        stage (charge=False: the retrain cost covers it). spark_throughput
        > 0 (legacy) makes a job take at least events / throughput ticks."""
        if self.spark_job is not None:
            job = self.spark_job
            when = ("finishes at tick %d" % job["done_tick"]) if job["status"] == "running" \
                else "waiting for the data lake"
            raise ActionError("a Spark job is already running (%s)" % when)
        if charge:
            self.budget_used += COST_SPARK
        self.spark_job = {"start": self.tick, "done_tick": None, "events": events, "by": by,
                          "then": then, "run": None, "status": "waiting"}
        if self.settings["spark_wait_for_data"] and not self._lake_complete():
            lake = self.system["datalake"]
            self._log("job", "Spark feature job (by %s) waiting for the data lake: it has data up to tick %d, "
                      "not %d." % (by, lake.ingested_tick, self.tick), cls="s-warn")
            return "spark job waiting for the data lake"
        return self._launch_spark()

    def _lake_complete(self):
        return self.tick == 0 or self.system["datalake"].ingested_tick >= self.tick

    def _check_spark_sensor(self):
        job = self.spark_job
        if not self.settings["spark_wait_for_data"] or self._lake_complete():
            self._launch_spark()
            return
        if self.tick - job["start"] >= self.settings["spark_sensor_timeout"]:
            self.spark_job = None
            lake = self.system["datalake"]
            error = ("sensor timed out after %d ticks: the data lake has data up to tick %d only"
                     % (self.tick - job["start"], lake.ingested_tick))
            self.spark_failures += 1
            self.last_spark_error = {"tick": self.tick, "reason": "sensor_timeout", "error": error, "by": job["by"]}
            self._log("job", "Spark feature job FAILED (by %s): %s" % (job["by"], error), cls="s-crit")
            if self.logger is not None:
                self.logger.log_event("spark_failed", self.tick, events=0, by=job["by"], reason="sensor_timeout")
            if job["then"] is not None:
                job["then"](False, None)

    def _launch_spark(self):
        job = self.spark_job
        lake = self.system["datalake"]
        if job["events"] is None:
            job["events"] = list(lake.events)
        events = job["events"]
        run = self.system["spark"].plan_offline_feature_job(events, self.offline.feature_pipeline,
                                                            lake=lake, submitted_tick=self.tick)
        duration = run.ticks
        throughput = self.settings["spark_throughput"]
        if throughput:
            duration = max(duration, max(1, -(-len(events) // throughput)))
        job.update(run=run, status="running", started=self.tick, done_tick=self.tick + duration)
        report = run.report
        if duration:
            self._log("job", "Spark feature job started (by %s): %s over %d events; finishes at tick %d."
                      % (job["by"], run.batch.reason, run.batch.size, self.tick + duration))
            return "spark job started, finishes at tick %d" % (self.tick + duration)
        return self._finish_spark()

    def _finish_spark(self):
        job, self.spark_job = self.spark_job, None
        events = job["events"]
        run = job["run"]
        try:
            self.system["spark"].run_offline_feature_job(events, self.offline.feature_pipeline, run=run)
            ok = True
        except RuntimeError as error:
            ok = False
            self.spark_failures += 1
            self.last_spark_error = {"tick": self.tick, "reason": run.report.reason, "error": run.report.error,
                                     "by": job["by"]}
            self._log("job", "Spark feature job FAILED (by %s): %s" % (job["by"], error), cls="s-crit")
            if self.logger is not None:
                self.logger.log_event("spark_failed", self.tick, events=len(events), by=job["by"],
                                      reason=run.report.reason)
        self._spark_bill += run.summary()["cost"]
        if ok:
            self.last_spark_tick = self.tick
            self.spark_runs += 1
            self.last_spark_error = None
            self._log("job", "Spark feature job finished (by %s): %s over %d events in %.1f min."
                      % (job["by"], run.batch.reason, run.batch.size, run.report.duration_s / 60.0))
            if self.logger is not None:
                self.logger.log_event("spark", self.tick, events=len(events), by=job["by"])
        if job["then"] is not None:
            job["then"](ok, events)
        return "spark job finished" if ok else "spark job failed"

    def _spark_view(self):
        spark = self.system["spark"]
        job = self.spark_job
        last = spark.last_run
        return {"last_run_tick": self.last_spark_tick, "runs": self.spark_runs,
                "failures": self.spark_failures, "last_error": self.last_spark_error,
                "every": self.settings["spark_every"],
                "running": None if job is None else {"start": job["start"], "done_tick": job["done_tick"],
                                                     "status": job["status"], "by": job["by"]},
                "cluster": spark.config.summary(), "price_per_hour": round(spark.config.price_per_hour(), 2),
                "incremental": spark.incremental, "tick_seconds": TICK_SECONDS,
                "last_job": None if last is None else last.summary(timeline=True),
                "history": [{k: h[k] for k in ("job_id", "mode", "reason", "events", "submitted_tick", "ticks",
                                               "status", "error", "duration_s", "cost")}
                            for h in spark.history]}

    # Retrain / deploy --------------------------------------------------

    def _start_retrain(self, by):
        if self.retrain is not None:
            raise ActionError("a training run is already in progress (deploys at tick %d)" % self.retrain["deploy_tick"])
        if self.budget_used + COST_RETRAIN > BUDGET_TOTAL:
            raise ActionError("not enough compute budget left for a retrain")
        if self.canary is not None:
            raise ActionError("a canary is running; promote or abort it first")
        if self.spark_job is not None:
            job = self.spark_job
            raise ActionError("a Spark job is running; retrain when it finishes (%s)"
                              % ("tick %d" % job["done_tick"] if job["status"] == "running"
                                 else "it is waiting for the data lake"))
        if not any(type(e).__name__ == "RecommendationClickedEvent" for e in self.system["datalake"].events):
            raise ActionError("no clicks to learn from yet; let the simulation run first")

        now = self.tick
        version = max(self.versions) + 1
        events = list(self.system["datalake"].events)

        self.budget_used += COST_RETRAIN
        record = {
            "version": version,
            "started_tick": now,
            "deploy_tick": now + self.settings["deploy_delay"],
            "stage": "features",
            "workflow": self._new_workflow(now, by),
            "cluster": training_job.TrainingCluster(
                gpus=self.settings["train_gpus"], gpu_memory_gb=self.settings["train_gpu_memory_gb"],
                batch_size=self.settings["train_batch_size"],
                checkpoint_minutes=self.settings["train_checkpoint_minutes"]),
            "batch_size": self.settings["train_batch_size"],
            "lr": float(self.settings["learning_rate"]),
            "validate_data": self.settings["validate_data"],
            "validate_model": self.settings["validate_model"],
            "train_report": None,
            "fail_at": None,
            "data_checks": None,
            "model_checks": None,
            "losses": None,
            "training_examples": None,
            "epochs": self.settings["epochs"],
            "negatives": self.settings["negatives"],
            "batched": self.settings["batched_training"],
            "rebuild_index": self.settings["rebuild_index"],
            "ranker": self.settings["ranker"],
            "ranker_mlp": self.settings["ranker_mlp"],
            "ranker_features": ("v6" if self.settings["ranker_signals"] and self.settings["ranker_tower_score"]
                                and self.settings["ranker_preference_score"] and self.settings["preference_model"] else
                                "v5" if self.settings["ranker_signals"] and self.settings["ranker_tower_score"] else
                                "v4" if self.settings["ranker_signals"] else
                                "v3" if self.settings["ranker_genre_features"] else "v2"),
            "ranker_multitask": self.settings["ranker_multitask"],
            "logq_correction": self.settings["logq_correction"],
            "random_negatives": self.settings["random_negatives"],
            "graded_positives": self.settings["graded_positives"],
            "tower_history": self.settings["tower_history"] or self.settings["tower_shared_ids"],
            "tower_engagement": self.settings["tower_engagement"],
            "tower_demographics": self.settings["tower_demographics"],
            "tower_shared_ids": self.settings["tower_shared_ids"],
            "train_from_scratch": self.settings["train_from_scratch"],
            "ranker_rows": None,
            "preferences": None,
            "cf_interactions": None,
            "cf_factors": self.settings["cf_factors"] if self.settings["cf_model"] else 0,
            "cf": None,
            "index_kind": self.index_kind,
            "ann_nprobe": self.settings["ann_nprobe"],
            "eval_gate": self.settings["eval_gate"],
            "holdout_pct": self.settings["holdout_pct"],
            "gate_tolerance_pct": self.settings["gate_tolerance_pct"],
            "holdout": None,
            "production": None,
            "recall": None,
            "by": by,
            "done": threading.Event(),
            "result": None,
            "error": None,
            "thread": None,
        }
        self.retrain = record
        if self.logger is not None:
            self.logger.log_event("retrain_start", now, model_version=version,
                                  deploy_tick=record["deploy_tick"])
        # Stage 1: offline features (Spark). Training starts when it ends.
        # events=None: the job reads the lake when it starts (the same events
        # unless it has to wait for the lake to catch up).
        self._run_spark(by="retrain v%d" % version, events=None, charge=False,
                        then=lambda ok, used: self._features_ready(record, used if used is not None else events, ok))
        if record["stage"] == "features":
            return "retrain v%d: waiting for the Spark job" % version
        return "training v%d, deploys at tick %d" % (version, record["deploy_tick"])

    def _features_ready(self, record, events, ok):
        """Stage 2, at a tick boundary: build the training set and start the
        background training. Everything that reads shared state or uses the
        session RNGs happens here: the training set (negative sampling,
        shuffle), the item features the new item tower will embed and the
        ranker's point-in-time rows."""
        version = record["version"]
        if not ok:
            self.retrain = None
            self._log("train", "Retrain v%d aborted: its Spark feature job failed." % version, cls="s-crit")
            self._step(record, "features", "failed", "the Spark feature job failed")
            self._end_workflow(record, "failed", "features")
            return
        self._step(record, "features", "succeeded",
                   "%s over %d events" % (self.system["spark"].last_run.batch.reason,
                                          self.system["spark"].last_run.batch.size)
                   if self.system["spark"].last_run is not None else None)
        train_events = events
        if record["eval_gate"]:
            # Hold out the newest events: train on the rest, score both
            # models on the held-out clicks (features as of each click).
            split = int(len(events) * (1 - record["holdout_pct"] / 100.0))
            train_events = events[:split]
            store = self.system["feature_store"]
            holdout = []
            for order, event in enumerate(events[split:], start=split):
                if type(event).__name__ != "RecommendationClickedEvent":
                    continue
                features = dict(store.get_user_features_as_of(event.user_id, event.timestamp, order)
                                or USER_FEATURE_DEFAULTS)
                features["user_id"] = event.user_id
                holdout.append((features, event.movie_id))
            record["holdout"] = holdout
            record["production"] = self.versions[self.serving_version]

        generator = self.offline.dataset_generator
        generator.negative_samples = record["negatives"]
        generator.graded = record["graded_positives"]
        generator.history = record["tower_history"]
        for order, event in enumerate(train_events):
            generator.add_event(event, order)
        training_data = generator.build()
        if not training_data:
            self.retrain = None
            self._log("train", "Retrain v%d aborted: no usable training examples." % version, cls="s-warn")
            self._step(record, "validate data", "failed", "no usable training examples")
            self._end_workflow(record, "failed", "validate data")
            return
        # Data validation: the training set against the last good one.
        reference = self.data_reference
        stats = validation.training_data_stats(len(training_data), train_events, self.system["feature_store"],
                                               since_tick=reference["through_tick"] if reference else None)
        checks = validation.validate_data(stats, reference)
        record["data_checks"] = {"stats": stats, "checks": checks, "reference": reference}
        failed = [c["name"] for c in checks if not c["ok"]]
        if failed and record["validate_data"]:
            self.retrain = None
            self._log("train", "Retrain v%d stopped by data validation: %s." % (version, ", ".join(failed)),
                      cls="s-crit")
            self._step(record, "validate data", "failed", "failed: " + ", ".join(failed))
            self._end_workflow(record, "failed", "validate data")
            return
        if not failed:
            self.data_reference = stats
        self._step(record, "validate data", "succeeded" if not failed else "skipped",
                   "%d examples, %s" % (stats["examples"], "all checks passed" if not failed
                                        else "checks failed but validation is off: " + ", ".join(failed)))
        items = _ItemFeatureSnapshot(self.offline.feature_pipeline.feature_store)
        if self.flink.preferences is not None:
            # Refit the preference model on everything it has logged so far
            # (a copy; the live one keeps learning online meanwhile).
            record["preferences"] = self.flink.preferences.snapshot()
        if record["cf_factors"]:
            # Who did what with which movie, as of now (the fit runs in the background).
            record["cf_interactions"] = ImplicitMF.interactions(train_events)
        if record["ranker"] and record["ranker_mlp"]:
            record["ranker_inputs"] = [] if record["ranker_features"] in ("v5", "v6") else None
            record["ranker_rows"] = MLPRanker.training_rows_from(
                train_events, self.system["feature_store"], self._genres_of, record["ranker_features"],
                with_completion=record["ranker_multitask"], inputs=record["ranker_inputs"])
            if record["ranker_features"] in ("v5", "v6"):
                # Join the logged serving-time scores (no leakage).
                inputs = record.pop("ranker_inputs")
                v6 = record["ranker_features"] == "v6"
                record["logged_share"] = MLPRanker.fill_tower_scores_from_log(
                    record["ranker_rows"][0], inputs, self.score_log, column=-2 if v6 else -1)
                if v6:
                    record["preference_logged_share"] = MLPRanker.fill_tower_scores_from_log(
                        record["ranker_rows"][0], inputs, self.preference_log)
        elif record["ranker"]:
            record["ranker_rows"] = LogisticRanker.training_rows_from(train_events, self.system["feature_store"])

        # A slow Spark job pushes the deploy back (deterministically): the
        # training stage always gets at least min(10, deploy_delay) ticks.
        if self.tick > record["started_tick"]:
            delay = record["deploy_tick"] - record["started_tick"]
            record["deploy_tick"] = max(record["deploy_tick"], self.tick + min(10, delay))
        record["training_examples"] = len(training_data)
        record["stage"] = "training"
        # The training job on the GPU cluster: simulated time and outcome.
        conditions = self.system["training_conditions"]
        rows = record["ranker_rows"][0] if record["ranker_rows"] is not None else []
        report = training_job.simulate_training(
            len(training_data), record["epochs"], record["negatives"] + record["random_negatives"],
            record["cluster"], conditions,
            training_job.rng_for(self.seed, version) if conditions.preemption_mean_s else None,
            items=len(items.item_features), extra_rows=len(rows))
        record["train_report"] = report
        record["train_ticks"] = ticks_for(report.duration_s)
        self._step(record, "train", "running")
        if not report.succeeded:
            record["fail_at"] = self.tick + record["train_ticks"]
            self._log("train", "Training v%d started on %d examples (%d GPU%s)."
                      % (version, len(training_data), record["cluster"].gpus,
                         "s" if record["cluster"].gpus > 1 else ""), cls="s-accent")
            if record["fail_at"] <= self.tick:
                self._training_failed(record)
            return
        # Model validation scores the new and the serving model on the same
        # new examples (a sample of the training set).
        record["eval_sample"] = training_data[:EVAL_SAMPLE]
        record["eval_production"] = self.versions[self.serving_version]
        record["train_done_tick"] = self.tick + record["train_ticks"]
        record["deploy_tick"] = max(record["deploy_tick"], record["train_done_tick"])
        thread = threading.Thread(
            target=self._train, args=(record, training_data, items),
            name="retrain-v%d" % version, daemon=True,
        )
        record["thread"] = thread
        thread.start()
        self._log("train", "Training v%d started on %d examples (%d epochs, %d negatives, by %s). Deploys at tick %d."
                  % (version, len(training_data), record["epochs"], record["negatives"], record["by"],
                     record["deploy_tick"]), cls="s-accent")

    def _train(self, record, training_data, items):
        """Background thread: train, embed every movie, build the index.
        Touches only the trainer (never served directly) and its inputs."""
        try:
            started = time.monotonic()
            trainer = self.offline.trainer
            current = getattr(trainer, "tower_signature", (False, False, False, False))
            if record["train_from_scratch"] or current != self._tower_signature(record):
                trainer = self.offline.trainer = self._fresh_trainer(record)
            trainer.epochs = record["epochs"]
            trainer.batched = record["batched"] or bool(
                record["logq_correction"] or record["random_negatives"] or record["graded_positives"])
            trainer.logq_correction = record["logq_correction"]
            trainer.random_negatives = record["random_negatives"]
            trainer.graded = record["graded_positives"]
            trainer.item_pool = [items.item_features[m] for m in sorted(items.item_features)]
            trainer.seed = self.seed
            trainer.batch_size = record["batch_size"]
            for group in trainer.optimizer.param_groups:
                group["lr"] = record["lr"]
            trainer.train(training_data)
            record["losses"] = [float(x) for x in getattr(trainer, "last_losses", [])]

            user_tower = copy.deepcopy(trainer.user_tower).eval()
            item_tower = copy.deepcopy(trainer.item_tower).eval()

            embeddings = {
                movie_id: item_tower.embed(features)
                for movie_id, features in items.item_features.items()
            }
            index = make_index(record["index_kind"], seed=self.seed * 1000 + record["version"],
                               nprobe=record["ann_nprobe"])
            index.build(embeddings)

            ranker = None
            try:
                if record["ranker_rows"] is not None and record["ranker_mlp"]:
                    rows, positions, labels, *rest = record["ranker_rows"]
                    record["ranker_rows"] = None
                    ranker = MLPRanker(self.retriever.user_features, self.system["redis"],
                                       self.feast, self._genres_of,
                                       seed=self.seed * 1000 + record["version"],
                                       feature_set=record["ranker_features"],
                                       multitask=record["ranker_multitask"])
                    if record["ranker_features"] in ("v5", "v6"):
                        ranker.set_towers(user_tower, embeddings)
                    if record["ranker_features"] == "v6":
                        ranker.preference_fn = lambda: self.flink.preferences
                        ranker.preference_logged_share = record.get("preference_logged_share")
                    ranker.fit(rows, positions, labels, rest[0] if rest else None)
                elif record["ranker_rows"] is not None:
                    rows, labels = record["ranker_rows"]
                    record["ranker_rows"] = None
                    ranker = LogisticRanker(self.retriever.user_features, self.feast)
                    ranker.fit(rows, labels)

            except ValueError as error:
                # Not enough clicks to fit a ranker yet: ship the towers.
                ranker = None
                record["ranker_error"] = str(error)
            if record["preferences"] is not None:
                record["preferences"].fit()
            if record["cf_interactions"] is not None:
                world = self.system["world"]
                record["cf"] = ImplicitMF(max(world.user_objects) + 1, max(world.movie_objects) + 1,
                                          factors=record["cf_factors"], seed=self.seed).fit(record["cf_interactions"])
                record["cf_interactions"] = None
            if record["holdout"]:
                production = record["production"]
                record["recall"] = (
                    recall_at_k(user_tower, index, record["holdout"]),
                    recall_at_k(production.user_tower, production.index, record["holdout"]),
                )

            eval_loss = production_loss = None
            sample = record.pop("eval_sample", None)
            if sample:
                production = record["eval_production"]
                eval_loss = evaluation_loss(user_tower, item_tower, sample, trainer.temperature)
                try:
                    production_loss = evaluation_loss(production.user_tower, production.item_tower, sample,
                                                      trainer.temperature)
                except Exception:      # the serving model takes other inputs (tower settings changed)
                    production_loss = None
            record["eval_losses"] = (eval_loss, production_loss)
            record["model_checks"] = validation.validate_model(record["losses"], embeddings,
                                                               eval_loss, production_loss)
            record["result"] = (user_tower, item_tower, index, ranker)
            record["train_seconds"] = round(time.monotonic() - started, 2)
        except Exception as error:  # surfaced in the log at deploy time
            record["error"] = repr(error)
        finally:
            record["done"].set()

    # Retrain workflow (an Airflow-style DAG run) -------------------------

    def _new_workflow(self, now, by):
        names = ["features", "validate data", "train", "validate model"]
        if self.settings["eval_gate"]:
            names.append("evaluate")
        names.append("deploy")
        return {"started_tick": now, "by": by, "status": "running", "failed_step": None,
                "steps": [{"name": n, "status": "pending", "start": None, "end": None, "detail": None}
                          for n in names]}

    def _step(self, record, name, status, detail=None, end=None):
        for step in record["workflow"]["steps"]:
            if step["name"] != name:
                continue
            if step["start"] is None:
                step["start"] = record["started_tick"] if name == "features" else self.tick
            if status in ("succeeded", "failed", "skipped"):
                step["end"] = end if end is not None else self.tick
            step["status"] = status
            if detail is not None:
                step["detail"] = detail

    def _end_workflow(self, record, status, failed_step=None):
        workflow = record["workflow"]
        workflow.update(status=status, failed_step=failed_step, ended_tick=self.tick)
        for step in workflow["steps"]:
            if step["status"] == "pending":
                step["status"] = "skipped" if status == "failed" else step["status"]
        report = record.get("train_report")
        run = {"version": record["version"], **workflow,
               "training": None if report is None else report.summary(),
               "train_ticks": record.get("train_ticks"),
               "cluster": record["cluster"].summary(), "lr": record["lr"],
               "data_checks": record.get("data_checks"), "model_checks": record.get("model_checks"),
               "losses": record.get("losses")}
        self.workflow_runs.append(run)
        del self.workflow_runs[:-20]
        self.last_workflow_failure = None if status == "succeeded" else run
        if self.logger is not None:
            self.logger.log_event("workflow", self.tick, model_version=record["version"], status=status,
                                  failed_step=failed_step)

    def _workflow_view(self):
        r = self.retrain
        current = None
        if r is not None:
            report = r["train_report"]
            current = {"version": r["version"], **r["workflow"], "deploy_tick": r["deploy_tick"],
                       "train_done_tick": r.get("train_done_tick"), "fail_at": r["fail_at"],
                       "training": None if report is None else report.summary(),
                       "data_checks": r["data_checks"]}
        runs = self.workflow_runs
        return {"current": current, "last": runs[-1] if runs else None,
                "history": [{k: run[k] for k in ("version", "by", "started_tick", "ended_tick", "status",
                                                 "failed_step", "train_ticks")} for run in runs],
                "gpu_price_per_hour": training_job.PRICE_PER_GPU_HOUR,
                "memory_gb": round(training_job.memory_gb(self.settings["train_batch_size"],
                                                          self.settings["negatives"]
                                                          + self.settings["random_negatives"]), 1),
                "missed_runs": len(self.missed_runs)}

    def _training_failed(self, record):
        """The simulated training job failed (GPU out of memory, too many
        preemptions): no model, the GPU time is still paid for."""
        report = record["train_report"]
        self.retrain = None
        self._train_bill += report.gpu_seconds / 3600.0 * training_job.PRICE_PER_GPU_HOUR
        self._step(record, "train", "failed", report.error)
        self._log("train", "Training v%d FAILED: %s." % (record["version"], report.error), cls="s-crit")
        self._end_workflow(record, "failed", "train")

    def _finish_retrain(self):
        record = self.retrain
        record["done"].wait(self.training_wait_timeout)
        record["thread"].join()
        self.retrain = None
        now = self.tick
        version = record["version"]

        report = record["train_report"]
        self._train_bill += report.gpu_seconds / 3600.0 * training_job.PRICE_PER_GPU_HOUR
        self._step(record, "train", "succeeded" if record["error"] is None else "failed",
                   record["error"] or "%d examples x %d epochs, %.0f min on %d GPU%s"
                   % (record["training_examples"], record["epochs"], report.duration_s / 60,
                      record["cluster"].gpus, "s" if record["cluster"].gpus > 1 else ""),
                   end=record.get("train_done_tick"))
        if record["error"] is not None:
            self._log("train", "Training v%d failed: %s" % (version, record["error"]), cls="s-crit")
            self._end_workflow(record, "failed", "train")
            return

        user_tower, item_tower, index, ranker = record["result"]
        gate = None
        if record["recall"] is not None:
            new_recall, prod_recall = record["recall"]
            passed = new_recall >= prod_recall * (1 - record["gate_tolerance_pct"] / 100.0)
            gate = {"recall_at_20": new_recall, "production_version": record["production"].version,
                    "production_recall_at_20": prod_recall, "holdout_clicks": len(record["holdout"]),
                    "passed": passed}
        self.versions[version] = ModelVersion(
            version, user_tower, item_tower, index, ranker=ranker, preferences=record["preferences"], cf=record["cf"],
            started_tick=record["started_tick"],
            trained_tick=record["started_tick"],
            deployed_tick=None,
            training_examples=record["training_examples"],
            epochs=record["epochs"],
            negatives=record["negatives"],
            train_seconds=record.get("train_seconds"),
            batched=record["batched"],
            recall_at_20=None if gate is None else gate["recall_at_20"],
            gate=gate,
            note="",
            preference_info=None if record["preferences"] is None else record["preferences"].summary(),
            ranker_info=None if ranker is None else {**ranker.summary(), "logged_share": record.get("logged_share")}
            if isinstance(ranker, MLPRanker) else {
                "model": "logistic",
                "training_rows": ranker.training_rows,
                "click_rate": round(ranker.positive_rate, 4),
                "weights": ranker.content_weights(),
            },
        )
        model_failed = [c["name"] for c in record["model_checks"] or [] if not c["ok"]]
        if model_failed and record["validate_model"]:
            self.versions[version].info["rejected"] = True
            self.versions[version].info["note"] = "failed model validation: " + ", ".join(model_failed)
            self._step(record, "validate model", "failed", "failed: " + ", ".join(model_failed))
            self._log("train", "v%d failed model validation (%s): not registered for deploy."
                      % (version, ", ".join(model_failed)), cls="s-crit")
            self._end_workflow(record, "failed", "validate model")
            return
        self._step(record, "validate model", "succeeded" if not model_failed else "skipped",
                   "all checks passed" if not model_failed
                   else "checks failed but validation is off: " + ", ".join(model_failed))
        if self.logger is not None:
            self.logger.log_event("retrain", now, model_version=version,
                                  training_examples=record["training_examples"])
        if record.get("ranker_error"):
            self._log("train", "v%d ships without a ranker: %s." % (version, record["ranker_error"]), cls="s-warn")
        self._log("train", "Training v%d finished (%d examples, %.1f s wall time)."
                  % (version, record["training_examples"], record.get("train_seconds") or 0), cls="s-accent")
        by = "auto" if record["by"] == "schedule" else record["by"]
        if gate is not None:
            self._log("train", "Gate: v%d Recall@20 %.1f%% vs v%d %.1f%% on %d held-out clicks: %s."
                      % (version, gate["recall_at_20"] * 100, gate["production_version"],
                         gate["production_recall_at_20"] * 100, gate["holdout_clicks"],
                         "passed" if gate["passed"] else "REJECTED, not deployed"),
                      cls="s-ok" if gate["passed"] else "s-warn")
            if self.logger is not None:
                self.logger.log_event("evaluation", now, model_version=version, k=20,
                                      recall_at_k=gate["recall_at_20"],
                                      production_recall_at_k=gate["production_recall_at_20"],
                                      passed=gate["passed"])
            self._step(record, "evaluate", "succeeded" if gate["passed"] else "failed",
                       "Recall@20 %.1f%% vs v%d %.1f%%" % (gate["recall_at_20"] * 100, gate["production_version"],
                                                          gate["production_recall_at_20"] * 100))
            if not gate["passed"]:
                self.versions[version].info["rejected"] = True
                self._end_workflow(record, "failed", "evaluate")
                return
        if self.settings["canary_share"] > 0:
            self._start_canary(version, by, record["rebuild_index"])
            self._step(record, "deploy", "succeeded", "canary to %d%% of users" % self.settings["canary_share"])
        else:
            self._deploy(version, by=by, rebuild_index=record["rebuild_index"])
            self._step(record, "deploy", "succeeded", "v%d serving" % version)
        self._end_workflow(record, "succeeded")

    def _deploy(self, version, by, rebuild_index=True, note=""):
        model = self.versions[version]
        self.retriever.user_tower = model.user_tower
        self.retriever.item_tower = model.item_tower
        self.cf_live = model.cf        # collaborative filtering is part of a version too
        if self.flink.preferences is not None:
            # The preference model is part of a version: deploying (or rolling
            # back to) a version restores its fitted copy, or the unfitted
            # state for a version that never had one (v0).
            if model.preferences is not None:
                self.flink.preferences.install(model.preferences)
            else:
                self.flink.preferences.unfit()
        self.model_server.ranker = self._serving_ranker(model)
        if rebuild_index:
            self.retriever.ann_index = model.index
            self.index_version = version
        previous = self.serving_version
        self.serving_version = version
        self.drift_reference = None      # rebuilt from this model's first clicks
        if model.info.get("deployed_tick") is None:
            model.info["deployed_tick"] = self.tick
        self.deploy_history.append((self.tick, version))

        if self.logger is not None:
            self.logger.model_version = version
            self.logger.log_event("deploy", self.tick, model_version=version,
                                  index_version=self.index_version, by=by)

        index_text = ("ANN index rebuilt with v%d embeddings" % version) if rebuild_index \
            else ("ANN index NOT rebuilt (still v%d)" % self.index_version)
        self._log("deploy", "Model v%d deployed by %s%s (was v%d). %s."
                  % (version, by, " (" + note + ")" if note else "", previous, index_text),
                  cls="s-accent" if by == "you" else "s-info")

        # Open CTR incidents are mitigated by a new deploy (component
        # incidents stay open until their component is healthy again).
        for incident in self.incidents:
            if incident["status"] == "Open" and incident["rule"] in ("deploy_ctr", "ctr_drop"):
                incident["status"] = "Mitigated"
                incident["timeline"].append((self.tick, "Model v%d deployed (by %s)." % (version, by)))
                self.open_rules.pop(incident["rule"], None)

        # Swapping in a freshly built index also replaces a broken one.
        if rebuild_index:
            self.faults.fix("rebuild_index", self.tick)

        self._watch_deploy(previous)

    def _export(self, version):
        if version not in self.versions:
            raise ActionError("no model version %r" % (version,))
        if self.system["run_dir"] is None:
            raise ActionError("this session has no run directory (metrics logging is off)")
        model = self.versions[version]
        directory = export_bundle(
            Path(self.system["run_dir"]) / "models" / ("v%d" % version),
            model.user_tower, model.item_tower, model.index, ranker=model.ranker,
            metadata={"version": version, "seed": self.seed, "exported_tick": self.tick, **model.info},
        )
        self._log("registry", "Model v%d exported to %s." % (version, directory), cls="s-accent")
        return str(directory)

    # Canary ------------------------------------------------------------

    def _start_canary(self, version, by, rebuild_index):
        model = self.versions[version]
        s = self.settings
        retriever = TwoTowerRetrieval(
            model.user_tower, model.item_tower,
            model.index if rebuild_index else self.retriever.ann_index,
            self.system["redis"], self.feast,
        )
        experiment = Experiment(self.serving_version, version, s["canary_share"],
                                self.tick + 1, self.tick + s["canary_ticks"], salt=version)

        def router(user_id):
            if experiment.arm(user_id) == "treatment":
                return retriever, self._serving_ranker(model)
            return self.retriever, self.model_server.ranker

        self.model_server.router = router
        self.canary = {"experiment": experiment, "rebuild_index": rebuild_index, "by": by}
        self._log("deploy", "Canary: v%d serves %d%% of users for %d ticks (v%d serves the rest)."
                  % (version, s["canary_share"], s["canary_ticks"], self.serving_version), cls="s-accent")
        if self.logger is not None:
            self.logger.log_event("canary_start", self.tick, model_version=version,
                                  share=s["canary_share"], until=experiment.end_tick)

    def _canary_tick(self):
        experiment = self.canary["experiment"]
        # Oldest first: a click must come after the slate it belongs to.
        for event in reversed(self._recent_events(1)):
            experiment.observe(event)
        if self.tick >= experiment.end_tick:
            test = experiment.stats()["test"]
            worse = test is not None and test["z"] < -1.96
            self._end_canary(promote=not worse, by="canary check")

    def _end_canary(self, promote, by):
        if self.canary is None:
            raise ActionError("no canary is running")
        canary, self.canary = self.canary, None
        experiment = canary["experiment"]
        stats = experiment.stats()
        self.model_server.router = None
        version = experiment.treatment
        self.versions[version].info["canary"] = {
            "share": experiment.share_pct, "start": experiment.start_tick, "end": self.tick,
            "control_ctr": stats["control"]["ctr"], "canary_ctr": stats["treatment"]["ctr"],
            "result": "promoted" if promote else "rolled back",
        }
        if self.logger is not None:
            self.logger.log_event("canary_end", self.tick, model_version=version, promoted=promote,
                                  control_ctr=stats["control"]["ctr"], canary_ctr=stats["treatment"]["ctr"])
        text = "Canary v%d %s (by %s): canary CTR %s vs %s on v%d." % (
            version, "promoted" if promote else "rolled back", by,
            _pct(stats["treatment"]["ctr"]), _pct(stats["control"]["ctr"]), experiment.control)
        self._log("deploy", text, cls="s-ok" if promote else "s-warn")
        if promote:
            self._deploy(version, by="canary" if by == "canary check" else by,
                         rebuild_index=canary["rebuild_index"])
        elif by == "canary check":
            incident = self._open("canary", "SEV2", "Canary v%d lost to v%d" % (version, experiment.control),
                                  text + " Rolled back automatically.")
            incident["status"] = "Mitigated"
            self.open_rules.pop("canary", None)
        return text

    def _rollback(self):
        if self.canary is not None:
            return self._end_canary(promote=False, by="you")
        if len(self.deploy_history) < 2:
            raise ActionError("nothing to roll back to")
        current = self.serving_version
        # The version that served before the current one.
        target = None
        for _, version in reversed(self.deploy_history):
            if version != current:
                target = version
                break
        if target is None:
            raise ActionError("nothing to roll back to")
        self._deploy(target, by="you", rebuild_index=True, note="rollback from v%d" % current)
        return "rolled back to v%d" % target

    # ------------------------------------------------------------------
    # Alert rule
    # ------------------------------------------------------------------

    def _watch_deploy(self, previous_version):
        s = self.settings
        tick = self.tick
        window = s["alert_window"]
        baseline = self.tracker.ctr_between(max(1, tick - window * 2 + 1), tick)
        self._watch = {
            "tick": tick,
            "version": self.serving_version,
            "previous": previous_version,
            "baseline": baseline,
            "until": tick + s["alert_ticks"],
        }

    def _check_alert(self):
        watch = self._watch
        if watch is None:
            return
        now = self.tick
        s = self.settings
        if now > watch["until"]:
            self._watch = None
            return
        window = s["alert_window"]
        # Wait until a full window after the deploy, so the rule looks only
        # at slates the new model served.
        if now - watch["tick"] < window:
            return
        current = self.tracker.ctr_between(now - window + 1, now)
        if ctr_drop_alert(watch["baseline"], current, s["alert_drop_pct"]):
            self._open_incident(watch, current)
            self._watch = None

    def _open_incident(self, watch, current):
        drop = (1 - current / watch["baseline"]) * 100
        incident = {
            "id": "INC-%d" % (len(self.incidents) + 1),
            "severity": "SEV2",
            "tick": self.tick,
            "status": "Open",
            "title": "Online CTR down %.0f%% since the model v%d deploy" % (drop, watch["version"]),
            "description": (
                "Rolling CTR over the last %d ticks is %.1f%%, down from %.1f%% before model v%d was "
                "deployed at tick %d (alert threshold: %d%% drop within %d ticks of a deploy)."
                % (self.settings["alert_window"], current * 100, watch["baseline"] * 100,
                   watch["version"], watch["tick"], self.settings["alert_drop_pct"],
                   self.settings["alert_ticks"])
            ),
            "rule": "deploy_ctr",
            "deploy_tick": watch["tick"],
            "version": watch["version"],
            "previous_version": watch["previous"],
            "baseline_ctr": watch["baseline"],
            "current_ctr": current,
            "runbook": list(RUNBOOK_CTR_DROP),
            "timeline": [
                (watch["tick"], "Model v%d deployed (was v%d)." % (watch["version"], watch["previous"])),
                (self.tick, "Alert fired: CTR %.1f%% vs %.1f%% before the deploy."
                 % (current * 100, watch["baseline"] * 100)),
            ],
        }
        self.incidents.append(incident)
        self.open_rules["deploy_ctr"] = incident
        self._log("alert", "%s opened: %s." % (incident["id"], incident["title"]), cls="s-crit")
        if self.logger is not None:
            self.logger.log_event("incident", self.tick, incident_id=incident["id"],
                                  model_version=watch["version"], baseline_ctr=watch["baseline"],
                                  current_ctr=current)

    # ------------------------------------------------------------------
    # Live settings, faults and repairs
    # ------------------------------------------------------------------

    def _apply_live_settings(self):
        s = self.settings
        self.model_server.slate_size = s["slate_size"]
        self.model_server.num_candidates = s["num_candidates"]
        if s["preference_model"] and self.flink.preferences is None:
            world = self.system["world"]
            self.flink.preferences = PreferenceModel(world.movie_objects, world.user_objects, seed=self.seed,
                                                     history=s["preference_history"],
                                                     shown_feature=s["preference_history"],
                                                     factors=s["preference_factors"])
            self.preference_source = PreferenceRetrieval(lambda: self.flink.preferences)
        if self.flink.preferences is not None:
            self.flink.preferences.cf_fn = self._cf_scores if s["cf_model"] else None
        if self.flink.preferences is not None:
            self.flink.preferences.half_life = s["preference_half_life"]
            self.flink.preferences.optimism = s["preference_optimism_pct"] / 100.0
            self.flink.preferences.taste_half_life = s["preference_taste_half_life"]
            self.flink.preferences.browse_decay = s["preference_browse_memory_pct"] / 100.0
            self.flink.preferences.grade_weight = s["preference_grade_weight_pct"] / 100.0
            self.flink.preferences.genre_optimism = s["preference_genre_optimism_pct"] / 100.0
            self.flink.preferences.use_history = s["preference_history"]
            self.flink.preferences.shown_feature = s["preference_history"]
        self.model_server.ranker = self._serving_ranker(self.versions[self.serving_version])
        chain = []
        if s["filter_watched"]:
            chain.append(WatchedFilter(self.system["redis"]))
        if s["diversity_pct"]:
            chain.append(MMRReranker(self._genres_of, diversity=s["diversity_pct"] / 100.0))
        if s["exploration_pct"]:
            if self.explorer is None or self.explorer.rate != s["exploration_pct"] / 100.0:
                self.explorer = ExplorationReranker(s["exploration_pct"] / 100.0,
                                                    lambda: self.model_server.slate_size, self.explore_rng)
            chain.append(self.explorer)
        if s["explore_slots"] and self.flink.preferences is not None:
            if getattr(self, "slot_explorer", None) is None:
                self.slot_explorer = SlotExplorer(lambda: self.flink.preferences, lambda: self.model_server.slate_size,
                                                  random.Random("explorer-%d" % self.seed), sim_fn=lambda: self.sim)
            e = self.slot_explorer
            e.slots, e.policy = s["explore_slots"], s["explore_policy"]
            e.rate, e.boosted_rate = s["explore_rate_pct"] / 100.0, s["explore_boost_pct"] / 100.0
            chain.append(e)
            self.flink.preferences.propensities = e.propensities if s["explore_ips"] else None
        elif self.flink.preferences is not None:
            self.flink.preferences.propensities = None
        self.model_server.reranker = RerankerChain(chain) if chain else None
        if (s["ranker"] and s["ranker_mlp"]) or s["tower_history"]:
            self.flink.track_recent = True     # recent history for the MLP ranker / user tower (stays on)
        if getattr(self, "profile_source", None) is None:
            # Second candidate source: content-based, from the genre profile.
            self.profile_source = ProfileRetrieval(self.retriever.user_features, self.feast,
                                                   self.system["world"].movie_objects)
        if getattr(self, "cowatch_source", None) is None:
            self.cowatch_source = CoWatchRetrieval(self.system["redis"])
        if getattr(self, "cf_source", None) is None:
            self.cf_source = CFRetrieval(lambda: self.cf_live, self.system["world"].movie_objects,
                                         lambda u, m: self.flink.preferences is not None
                                         and (u, m) in self.flink.preferences.watched)
        self.model_server.extra_sources = (
            ([(self.preference_source, s["preference_candidates"])]
             if s["preference_model"] and s["preference_candidates"] else [])
            + ([(self.cf_source, s["cf_candidates"])] if s["cf_model"] and s["cf_candidates"] else [])
            + ([(self.profile_source, s["profile_candidates"])] if s["profile_candidates"] else [])
            + ([(self.cowatch_source, s["cowatch_candidates"])] if s["cowatch_candidates"] else []))
        if s["tower_engagement"]:
            self.flink.track_engagement = True       # stays on once switched on
        self.offline.feature_pipeline.engagement = s["tower_engagement"]
        if s["cowatch_candidates"] or (s["ranker"] and s["ranker_mlp"] and s["ranker_signals"]):
            self.flink.track_item_signals = True   # stays on once switched on
        mode = "rich" if s["rich_profiles"] else "watches"
        self.flink.profile_mode = mode
        self.offline.feature_pipeline.profile_mode = mode
        spark = self.system["spark"]
        spark.config = ClusterConfig(
            executors=s["spark_executors"], cores=s["spark_executor_cores"],
            memory_gb=s["spark_executor_memory_gb"], shuffle_partitions=s["spark_shuffle_partitions"],
            aqe=s["spark_aqe"], speculation=s["spark_speculation"],
            max_task_failures=s["spark_task_max_failures"])
        spark.incremental = s["spark_incremental"]
        self.dynamics.viral_every = s["viral_every"]
        self.dynamics.viral_duration = s["viral_duration"]
        self.system["world"].onboarding = {"genres": 3, "noise": 0.3} if s["onboarding"] else None
        self.system["world"].browsing = {"rate": s["browse_pct"] / 100.0} if s["browse_pct"] else None
        self.simulator.rating_probability = s["rating_pct"] / 100.0
        if s["interest_save_pct"] or s["interest_dismiss_pct"]:
            if self.simulator.interest_rng is None:
                self.simulator.interest_rng = random.Random("interest-%d" % self.seed)
            self.simulator.interest = {"save": s["interest_save_pct"] / 100.0,
                                       "dismiss": s["interest_dismiss_pct"] / 100.0}
            self.flink.track_interest = True       # stays on once switched on
        else:
            self.simulator.interest = None
        self.system["world"].session_slates = s["session_slates"]
        if s["browse_pct"]:
            self.flink.track_browsing = True       # stays on once switched on
        if s["onboarding"]:
            self.flink.track_onboarding = True    # stays on once switched on
        self.flink.max_events_per_tick = s["flink_capacity"] or None
        self._apply_infra_settings()
        d = self.dynamics
        d.taste_drift_every = s["taste_drift_every"]
        d.taste_drift_share = s["taste_drift_share_pct"] / 100.0
        d.taste_drift_mode = s["taste_drift_mode"]
        d.users_joining_per_tick = s["users_joining_per_100_ticks"] / 100.0
        self.simulator.fatigue = s["fatigue_pct"] / 100.0
        self.simulator.fatigue_recovery = s["fatigue_recovery_ticks"]
        self.simulator.rewatch = s["rewatch_pct"] / 100.0
        if s["chaos_every"] and self.next_chaos_tick is None:
            self.next_chaos_tick = self.tick + self._chaos_gap()
        if not s["chaos_every"]:
            self.next_chaos_tick = None

    def _cf_scores(self, user_id, movies):
        cf = getattr(self, "cf_live", None)
        if cf is None:
            return np.zeros(len(movies))
        return cf.scores(user_id, movies)

    def _serving_ranker(self, model):
        """The ranker a model version serves with: the preference model when
        it is on (the version's own ranker until its first fit)."""
        own = model.ranker if self.settings["ranker"] else None
        if (own is not None and getattr(own, "feature_set", None) == "v6"
                and (getattr(own, "preference_logged_share", 0) or 0) >= 0.5):
            return own          # the MLP already combines the preference score with everything else
        if self.settings["preference_model"] and self.flink.preferences is not None:
            return PreferenceRanker(lambda: self.flink.preferences, fallback=own)
        return own

    def _apply_infra_settings(self):
        """Kafka, Flink, Redis, Feast, KServe and the ANN index, from the
        settings. Changes after the start that need work in a real system
        cost money once (ONE_OFF) and some take time (a Flink rescale)."""
        s, now = self.settings, self.tick
        first = not self._infra_applied
        kafka = self.kafka
        if not first and s["kafka_partitions"] != kafka.partitions:
            self.economy.charge(now, ONE_OFF["kafka_repartition"], "Kafka repartition to %d" % s["kafka_partitions"])
        kafka.partitions = s["kafka_partitions"]
        kafka.brokers = s["kafka_brokers"]
        kafka.replication_factor = s["kafka_replication"]
        kafka.retention_ticks = s["kafka_retention_ticks"]

        flink = self.flink
        flink.slot_capacity = s["flink_slot_capacity"]
        if first:
            flink.parallelism = s["flink_parallelism"]
        elif s["flink_parallelism"] != flink.parallelism:
            self.economy.charge(now, ONE_OFF["flink_rescale"], "Flink rescale to %d" % s["flink_parallelism"])
            flink.rescale(s["flink_parallelism"], now)
            self._log("config", "Flink rescaling to parallelism %d: savepoint + restart, back at tick %d."
                      % (s["flink_parallelism"], flink.restoring_until), cls="s-warn")

        redis = self.system["redis"]
        redis.max_keys = s["redis_max_keys"]
        redis.eviction = s["redis_eviction"]
        redis.ttl_ticks = {f: s["redis_ttl_ticks"] for f in TTL_FEATURES} if s["redis_ttl_ticks"] else {}
        redis.enforce()

        self.feast.mode = s["feast_mode"]
        self.feast.materialize_every = s["feast_materialize_every"]

        k = self.kserve
        k.min_replicas, k.max_replicas = s["kserve_min_replicas"], s["kserve_max_replicas"]
        k.replica_capacity = s["kserve_replica_capacity"]
        k.target_utilization = s["kserve_target_pct"] / 100.0
        k.cold_start_ticks = s["kserve_cold_start_ticks"]
        if first:
            k.ready = k.min_replicas
        k.ready = max(min(k.ready, k.max_replicas), 0)

        for model in self.versions.values():
            base = model.index
            while hasattr(base, "inner"):
                base = base.inner
            if hasattr(base, "nprobe"):
                base.nprobe = s["ann_nprobe"]
        self._infra_applied = True

    def _swap(self, component, to):
        """Swap a component for another that does the same job (now: the ANN
        index). Costs money once; the serving model's index is rebuilt with
        the new type from its own movie vectors, and new models use it."""
        if component != "ann_index":
            raise ActionError("only the ANN index can be swapped so far")
        if to == self.index_kind:
            raise ActionError("the ANN index is already %s" % INDEX_TYPES[to]["name"])
        cost = ONE_OFF["swap_ann_index"]
        if self.economy.cash < cost:
            raise ActionError("not enough cash for the migration ($%.0f)" % cost)
        model = self.versions[self.serving_version]
        started = time.monotonic()
        index = make_index(to, seed=self.seed * 1000 + model.version, nprobe=self.settings["ann_nprobe"])
        index.build(index_embeddings(model.index))
        seconds = time.monotonic() - started
        previous = self.index_kind
        self.economy.charge(self.tick, cost, "Swap ANN index %s -> %s" % (previous, to))
        model.index = index
        self.retriever.ann_index = index
        self.index_version = model.version
        self.golden.pop(model.version, None)        # new golden queries for the new index
        self.faults.fix("rebuild_index", self.tick)
        self.index_kind = to
        self._log("deploy", "ANN index swapped: %s -> %s (rebuilt for v%d in %.1f s, $%.0f). New models use %s."
                  % (INDEX_TYPES[previous]["name"], INDEX_TYPES[to]["name"], model.version, seconds, cost,
                     INDEX_TYPES[to]["name"]), cls="s-accent")
        if self.logger is not None:
            self.logger.log_event("swap", self.tick, component=component, to=to, previous=previous, cost=cost)
        return "ANN index is now %s" % INDEX_TYPES[to]["name"]

    def _account(self):
        """Money for the tick just run: revenue from its clicks, running
        costs of the current configuration, Feast materialization runs."""
        now = self.tick
        # Clicks users made this tick (published to Kafka), not what the
        # metrics consumer has read: a stalled consumer must not stop revenue.
        published = self.kafka.published.get("RecommendationClickedEvent", 0)
        clicks = published - getattr(self, "_clicks_billed", 0)
        self._clicks_billed = published
        breakdown = self.economy.running_costs(self.system, self.settings, self.index_kind)
        runs = self.feast.materializations - self._materializations_billed
        if runs:
            breakdown["Feast"] += runs * PRICES["feast_materialization"]
            self._materializations_billed = self.feast.materializations
        breakdown["Spark"] = self._spark_bill
        self._spark_bill = 0.0
        breakdown["Training"] = self._train_bill
        self._train_bill = 0.0
        self.economy.tick(now, clicks, breakdown)

    def _chaos_gap(self):
        every = self.settings["chaos_every"]
        return max(10, int(every * self.chaos_rng.uniform(0.5, 1.5)))

    def _chaos(self, now):
        """Hidden random faults on a seeded schedule (deterministic)."""
        if self.next_chaos_tick is None or now < self.next_chaos_tick:
            return
        kind = self.chaos_rng.choice(sorted(CHAOS_DURATIONS))
        self.faults.schedule(kind, now + 1, CHAOS_DURATIONS[kind])
        self.faults.apply_due(now)
        self.next_chaos_tick = now + self._chaos_gap()

    def _drill(self, kind, duration=None):
        if duration is not None and (not isinstance(duration, int) or duration < 1):
            raise ActionError("duration must be a positive whole number of ticks")
        fault = self.faults.schedule(kind, self.tick + 1, duration)
        self.faults.apply_due(self.tick)
        self.drills.add(fault.id)
        self._log("drill", "Drill: %s injected (%s)." % (FAULT_KINDS[kind],
                  "until fixed" if duration is None else "%d ticks" % duration), cls="s-warn")
        return "fault %s scheduled" % fault.id

    def _fix(self, action):
        if action == "rebuild_index":
            # Re-point serving at the serving version's own index (what a
            # rebuild with the serving item tower produces).
            model = self.versions[self.serving_version]
            self.retriever.ann_index = model.index
            self.index_version = self.serving_version
        ended = self.faults.fix(action, self.tick)
        if action == "restart_flink":
            self.flink.paused = False
        if action == "restart_model_server":
            self.model_server.retriever_down = False
        if action == "restart_kafka_broker":
            self.kafka.broker_up()
        if action == "restart_lake_sink":
            self.system["datalake"].paused = False
        if action == "restart_scheduler":
            self.system["scheduler"]["up"] = True
            self.missed_runs = []
        if action == "backfill_features":
            self.offline.feature_pipeline.null_profile_share = 0.0
            self.system["spark"].force_full = True
            if self.spark_job is None:
                self._run_spark(by="backfill")
        label = {"restart_flink": "Flink restarted", "rebuild_index": "ANN index rebuilt with v%d embeddings"
                 % self.serving_version, "restart_model_server": "Model server restarted",
                 "restart_kafka_broker": "Kafka broker restarted",
                 "replace_spark_node": "Degraded Spark node drained and replaced",
                 "restart_lake_sink": "Data lake sink restarted",
                 "restart_scheduler": "Workflow scheduler restarted",
                 "backfill_features": "Feature job fixed; full feature backfill started"}[action]
        self._log("ops", "%s by you." % label, cls="s-accent")
        if self.logger is not None:
            self.logger.log_event("repair", self.tick, action=action, faults_ended=[f.id for f in ended])
        return label

    # ------------------------------------------------------------------
    # Monitoring rules (symptom-based, like a real on-call setup)
    # ------------------------------------------------------------------

    def _ctr_range(self, start, end):
        shown = clicks = 0
        for tick in range(max(1, start), end + 1):
            c = self.tracker.counts.get(tick)
            if c:
                shown += c[0]
                clicks += c[1]
        return clicks / shown if shown else None

    def _recent_events(self, ticks):
        """Events of the last `ticks` ticks, newest first."""
        events = self.system["datalake"].events
        cutoff = self.tick - ticks
        out = []
        for event in reversed(events):
            if event.timestamp <= cutoff:
                break
            out.append(event)
        return out

    def _monitor(self):
        now = self.tick
        s = self.settings
        h = self.health

        # Every tick: lag and fallback share.
        h["flink_lag"] = self.kafka.lag("Flink")
        recent = [self.fallback_ticks.get(t, (0, 0)) for t in range(now - 9, now + 1)]
        requests = sum(r for r, _ in recent)
        h["fallback_share"] = (sum(f for _, f in recent) / requests) if requests else 0.0

        self._rule("flink_lag", h["flink_lag"] > s["lag_alert"], h["flink_lag"] < s["lag_alert"] / 4,
                   "WARN", "Flink consumer lag at %d events" % h["flink_lag"],
                   "Flink is %d events behind Kafka (alert at %d). Online features are going stale."
                   % (h["flink_lag"], s["lag_alert"]))
        self._rule("fallback", h["fallback_share"] > 0.5, h["fallback_share"] == 0,
                   "SEV2", "Model server serving popularity fallback",
                   "%.0f%% of requests in the last 10 ticks were served by the fallback, not the model."
                   % (h["fallback_share"] * 100))

        failure = self.last_workflow_failure
        self._rule("workflow", failure is not None, failure is None, "WARN",
                   "Retrain workflow failed at %s" % (failure["failed_step"] if failure else ""),
                   "Retrain v%d (by %s, started tick %d) failed at step '%s'."
                   % (failure["version"], failure["by"], failure["started_tick"], failure["failed_step"])
                   if failure else "")
        missed = self.missed_runs
        self._rule("schedule_missed", bool(missed), not missed, "WARN",
                   "Scheduled runs missed: %d" % len(missed),
                   "Scheduled runs did not start: %s." % ", ".join("%s at tick %d" % (k, t) for t, k in missed[-5:])
                   if missed else "")
        error = self.last_spark_error
        self._rule("spark_job", error is not None, error is None, "WARN",
                   "Spark feature job failed%s" % (" (%s)" % error["reason"] if error and error["reason"] else ""),
                   "The last Spark feature job (by %s, tick %d) failed: %s" % (error["by"], error["tick"], error["error"])
                   if error else "")
        last = self.system["spark"].last_run
        slow = last is not None and last.report.succeeded and last.ticks >= 1
        self._rule("spark_slow", slow, last is not None and last.ticks == 0, "WARN", "Spark job over a tick: %.0f min" % (last.report.duration_s / 60.0 if last else 0),
                   "The last Spark job took %.0f minutes (%d ticks of 15 minutes)."
                   % ((last.report.duration_s / 60.0, last.ticks) if last else (0, 0)))
        lake = self.system["datalake"]
        behind = now - lake.ingested_tick if now > 0 else 0
        self._rule("lake_ingestion", behind > 5, behind <= 1, "WARN",
                   "Data lake %d ticks behind" % behind,
                   "The lake sink has written no data since tick %d; %d events are waiting in Kafka."
                   % (lake.ingested_tick, self.kafka.lag("DataLake")))

        cash = self.economy.cash
        self._rule("budget", cash < 0, cash >= 0, "SEV2", "Over budget: cash $%.0f" % cash,
                   "Running costs and one-off charges exceed what clicks have earned. Cut infrastructure "
                   "(replicas, slots, brokers, a cheaper index) or improve CTR.")
        kafka_down = self.kafka.unavailable_partitions()
        self._rule("kafka_partitions", bool(kafka_down), not kafka_down, "SEV2",
                   "Kafka partitions offline: %s" % kafka_down,
                   "Partitions %s have no live replica: consumers (Flink, metrics) stop at them and lag "
                   "builds up." % kafka_down)
        k = self.kserve
        recent = k.last["requests"]
        share = k.last["overflow"] / recent if recent else 0.0
        h["serving_overflow"] = share
        self._rule("serving_capacity", share > 0.1, share == 0, "WARN",
                   "Model serving over capacity: %.0f%% overflow" % (share * 100),
                   "%.0f%% of last tick's requests found no free KServe replica and got the popularity "
                   "fallback." % (share * 100))

        if now % 10 == 0:
            self._ctr_rule(now)

        if now % 25 == 0:
            h["index_self_test"] = health.index_self_test(self.retriever.ann_index, self._golden())
            ok = h["index_self_test"]
            self._rule("index_health", ok is not None and ok < 0.8, ok is not None and ok >= 0.95,
                       "SEV2", "ANN index self-test failing",
                       "Only %.0f%% of the golden queries recorded when index v%d was built still return "
                       "the same movies (healthy is 100%%, alert below 80%%)." % ((ok or 0) * 100, self.index_version))

            events = self._recent_events(50)
            users = sorted({e.user_id for e in events})
            h["feature_skew"] = health.feature_skew(self.system["redis"], self.system["feature_store"], users)
            skew, limit = h["feature_skew"], s["skew_alert_pct"] / 100.0
            self._rule("feature_skew", skew is not None and skew > limit,
                       skew is not None and skew < limit * 0.7,
                       "WARN", "Online/offline feature skew %.0f%%" % ((skew or 0) * 100),
                       "Online genre preferences of recently served users differ from the offline ones by "
                       "%.0f%% on average (alert at %d%%)." % ((skew or 0) * 100, s["skew_alert_pct"]))

            shown = [m for e in self._recent_events(100) if hasattr(e, "movie_ids") for m in e.movie_ids]
            catalog = self.system["world"].movie_objects
            h["coverage"] = len(set(shown)) / len(catalog) if shown else None

            last = self.last_spark_tick if self.last_spark_tick is not None else 0
            age = now - last
            h["feature_age"] = age
            self._rule("freshness", age > s["freshness_alert"], age <= s["freshness_alert"],
                       "WARN", "Offline features %d ticks old" % age,
                       "The last successful Spark feature job was at tick %d (alert after %d ticks)."
                       % (last, s["freshness_alert"]))

            self._update_drift(now, catalog)
            psi = h["drift_psi"]
            self._rule("drift", psi is not None and psi > 0.2, psi is not None and psi < 0.1,
                       "WARN", "Data drift: click mix PSI %.2f" % (psi or 0),
                       "The genre mix of recent clicks has shifted from the model's training data "
                       "(PSI %.2f, alert at 0.2)." % (psi or 0))

    DRIFT_WINDOW = 200

    def _update_drift(self, now, catalog):
        """PSI of the genre mix of clicks in the last DRIFT_WINDOW ticks vs
        the first DRIFT_WINDOW ticks after the current model's deploy."""
        deploy_tick = self.deploy_history[-1][0]
        window = self.DRIFT_WINDOW
        if now < deploy_tick + window:
            self.health["drift_psi"] = None
            return

        def clicks(start, end):
            return [e.movie_id for e in self.system["datalake"].events
                    if start <= e.timestamp <= end and type(e).__name__ == "RecommendationClickedEvent"]

        if self.drift_reference is None:
            self.drift_reference = health.genre_distribution(clicks(deploy_tick + 1, deploy_tick + window), catalog)
        if now < deploy_tick + 2 * window:
            self.health["drift_psi"] = None
            return
        recent = [e.movie_id for e in self._recent_events(window)
                  if type(e).__name__ == "RecommendationClickedEvent"]
        self.health["drift_psi"] = health.genre_psi(self.drift_reference,
                                                    health.genre_distribution(recent, catalog))

    @staticmethod
    def _tower_signature(record):
        return (record["tower_history"], record["tower_engagement"], record["tower_demographics"],
                record["tower_shared_ids"])

    def _fresh_trainer(self, record):
        """New towers (and optimizer) for this retrain, seeded from the
        session seed and the version, with or without the history input."""
        import torch

        old = self.offline.trainer
        user_kwargs = dict(self.system["user_tower_kwargs"])
        num_movies = len(self.system["world"].movie_objects)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed * 1000 + record["version"])
            item_kwargs = dict(self.system["item_tower_kwargs"])
            if record["tower_shared_ids"]:
                item_kwargs.update(use_id_embedding=True, id_init_std=0.01)
            item = SimpleItemTower(num_movies=num_movies, **item_kwargs)
            demographics = None
            if record["tower_demographics"]:
                from ml_system.demographics import demographics_table
                demographics = demographics_table(self.system["world"].user_objects)
            user = SimpleUserTower(num_users=len(self.system["world"].user_objects),
                                   use_history=record["tower_history"], num_movies=num_movies,
                                   use_engagement=record["tower_engagement"], demographics=demographics,
                                   history_table=item.movie_embedding if record["tower_shared_ids"] else None,
                                   **user_kwargs)
        trainer = TwoTowerTrainer(user, item, epochs=old.epochs, temperature=old.temperature,
                                  batch_size=old.batch_size, batched=old.batched)
        trainer.tower_signature = self._tower_signature(record)
        return trainer

    def _log_scores(self, tick, user, slate, retriever):
        """Feature logging at serving time: user vector . movie vector of the
        retriever that served this slate (not the fallback), and the live
        preference model's score once it has been fitted (v6 ranker)."""
        model = self.flink.preferences
        if model is not None and model.fits:
            for movie_id, score in zip(slate, model.scores(user.user_id, list(slate))):
                self.preference_log[(tick, user.user_id, movie_id)] = float(score)
        user_tower = getattr(retriever, "user_tower", None)
        index = getattr(retriever, "ann_index", None)
        if user_tower is None or index is None:
            return
        base = index
        while hasattr(base, "inner"):
            base = base.inner
        nodes = getattr(base, "nodes", None)
        if not nodes:
            return
        vector = user_tower.embed(retriever.user_features(user.user_id))
        for movie_id in slate:
            node = nodes.get(movie_id)
            if node is not None:
                self.score_log[(tick, user.user_id, movie_id)] = sum(a * b for a, b in zip(vector, node.unit))

    def _genres_of(self, movie_id):
        return self.system["world"].movie_objects[movie_id].genres

    def _golden(self):
        """Golden queries for the index version that should be serving,
        recorded once from that version's own (freshly built) index."""
        version = self.index_version
        if version not in self.golden:
            self.golden[version] = health.golden_queries(self.versions[version].index,
                                                         self.system["world"].movie_objects)
        return self.golden[version]

    def _ctr_rule(self, now):
        s = self.settings
        window = s["alert_window"]
        if now < 5 * window:
            return
        current = self._ctr_range(now - window + 1, now)
        incident = self.open_rules.get("ctr_drop")
        if incident is not None:
            if current is not None and current >= incident["baseline_ctr"] * (1 - s["alert_drop_pct"] / 200.0):
                self._resolve(incident, "CTR back to %.1f%%." % (current * 100))
            return
        if "deploy_ctr" in self.open_rules:
            return
        baseline = self._ctr_range(now - 5 * window + 1, now - window)
        if ctr_drop_alert(baseline, current, s["alert_drop_pct"]):
            drop = (1 - current / baseline) * 100
            self._open("ctr_drop", "SEV2", "Online CTR down %.0f%%" % drop,
                       "Rolling CTR over the last %d ticks is %.1f%%, down from %.1f%% over the %d ticks "
                       "before (alert at a %d%% drop)." % (window, current * 100, baseline * 100,
                                                          4 * window, s["alert_drop_pct"]),
                       baseline_ctr=baseline, current_ctr=current)

    def _rule(self, rule, firing, clear, severity, title, description):
        incident = self.open_rules.get(rule)
        if incident is None and firing:
            self._open(rule, severity, title, description)
        elif incident is not None and clear:
            self._resolve(incident, "Condition cleared.")

    def _open(self, rule, severity, title, description, **extra):
        incident = {
            "id": "INC-%d" % (len(self.incidents) + 1),
            "rule": rule,
            "severity": severity,
            "tick": self.tick,
            "status": "Open",
            "title": title,
            "description": description,
            "deploy_tick": self.tick,
            "version": None,
            "previous_version": None,
            "runbook": list(RUNBOOKS[rule]),
            "timeline": [(self.tick, "Alert fired: %s." % title)],
            **extra,
        }
        self.incidents.append(incident)
        self.open_rules[rule] = incident
        self._log("alert", "%s opened: %s." % (incident["id"], title),
                  cls="s-crit" if severity == "SEV2" else "s-warn")
        if self.logger is not None:
            self.logger.log_event("incident", self.tick, incident_id=incident["id"], rule=rule)
        return incident

    def _resolve(self, incident, text):
        incident["status"] = "Resolved"
        incident["timeline"].append((self.tick, "Resolved: %s" % text))
        self.open_rules.pop(incident["rule"], None)
        self._log("alert", "%s resolved: %s" % (incident["id"], text), cls="s-ok")

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    def _wrap_recommend(self):
        original = self.model_server.recommend
        latency = self.latency
        clock = self.sim.clock

        def timed_recommend(user):
            started = time.perf_counter()
            try:
                return original(user)
            finally:
                latency.setdefault(clock.now, []).append((time.perf_counter() - started) * 1000)

        # Instance attribute: the World looks the method up on the service.
        self.model_server.recommend = timed_recommend

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log(self, kind, text, cls="s-info"):
        self.log.append({"seq": len(self.log) + 1, "tick": self.tick, "kind": kind, "cls": cls, "text": text})

    # ------------------------------------------------------------------
    # Read-only views (JSON-ready dicts for the API)
    # ------------------------------------------------------------------

    def state(self):
        with self.lock:
            r = self.retrain
            return {
                "tick": self.tick,
                "playing": self.playing,
                "speed": self.speed,
                "speeds": list(SPEEDS),
                "ticks_per_second": BASE_TICKS_PER_SECOND * self.speed,
                "waiting_for_training": self.waiting_for_training,
                "clock_error": self.clock_error,
                "seed": self.seed,
                "users_per_tick": self.users_per_tick,
                "num_users": len(self.system["world"].user_objects),
                "num_movies": len(self.system["world"].movie_objects),
                "model_version": self.serving_version,
                "index_version": self.index_version,
                "spark_job": None if self.spark_job is None else {
                    "start": self.spark_job["start"], "done_tick": self.spark_job["done_tick"],
                    "status": self.spark_job["status"],
                    "events": len(self.spark_job["events"] if self.spark_job["events"] is not None
                                  else self.system["datalake"].events),
                    "by": self.spark_job["by"]},
                "training": None if r is None else {
                    "stage": r["stage"],
                    "version": r["version"],
                    "started_tick": r["started_tick"],
                    "deploy_tick": r["deploy_tick"],
                    "done": r["done"].is_set(),
                },
                "budget": {"used": self.budget_used, "total": BUDGET_TOTAL,
                           "retrain_cost": COST_RETRAIN, "spark_cost": COST_SPARK},
                "money": {"cash": round(self.economy.cash, 2), "start_cash": self.economy.start_cash,
                          "last_tick_revenue": self.economy.last_tick["revenue"],
                          "last_tick_cost": self.economy.last_tick["cost"]},
                "ann_index": self.index_kind,
                "reference": self.reference,
                "run_dir": str(self.system["run_dir"]) if self.system["run_dir"] else None,
                "open_incidents": sum(1 for i in self.incidents if i["status"] == "Open"),
                "canary": self._canary_view(),
                "world": self.world_kind,
                "taste": self.taste_kind,
                "preset": self.preset,
                "health": dict(self.health),
                "pending_actions": len(self._pending),
            }

    def metrics(self, since=0):
        """Per-tick rows for ticks > since (up to the current tick)."""
        with self.lock:
            rows = []
            for tick in range(max(1, int(since) + 1), self.tick + 1):
                shown, clicks = self.tracker.counts.get(tick, (0, 0))
                ms = self.latency.get(tick)
                rows.append({
                    "tick": tick,
                    "shown": shown,
                    "clicks": clicks,
                    "ctr": clicks / shown if shown else None,
                    "active_users": len(self.tracker.users.get(tick, ())),
                    "model_version": self.tick_versions.get(tick),
                    "latency_p50_ms": round(statistics.median(ms), 2) if ms else None,
                    "latency_max_ms": round(max(ms), 2) if ms else None,
                })
            return {"tick": self.tick, "rows": rows,
                    "deploys": [{"tick": t, "version": v} for t, v in self.deploy_history]}

    def events(self, since=0):
        with self.lock:
            return [e for e in self.log if e["seq"] > int(since)]

    def models(self):
        with self.lock:
            out = []
            for version in sorted(self.versions, reverse=True):
                model = self.versions[version]
                deployed = [t for t, v in self.deploy_history if v == version]
                served_ticks = [t for t, v in self.tick_versions.items() if v == version]
                ctr = None
                if served_ticks:
                    shown = sum(self.tracker.counts.get(t, (0, 0))[0] for t in served_ticks)
                    clicks = sum(self.tracker.counts.get(t, (0, 0))[1] for t in served_ticks)
                    ctr = clicks / shown if shown else None
                out.append({
                    **model.summary(),
                    "deploys": deployed,
                    "ticks_served": len(served_ticks),
                    "online_ctr": ctr,
                    "stage": "Production" if version == self.serving_version else (
                        "Canary" if self.canary and self.canary["experiment"].treatment == version else
                        "Rejected" if model.info.get("rejected") else "Archived"),
                })
            r = self.retrain
            training = None
            if r is not None:
                training = {k: r[k] for k in ("version", "stage", "started_tick", "deploy_tick",
                                              "training_examples", "epochs", "negatives", "by")}
                training["done"] = r["done"].is_set()
            return {
                "canary": self._canary_view(),
                "serving_version": self.serving_version,
                "index_version": self.index_version,
                "rollback_target": self._rollback_target(),
                "training": training,
                "versions": out,
            }

    def _canary_view(self):
        if self.canary is None:
            return None
        e = self.canary["experiment"]
        return {"version": e.treatment, "control": e.control, "share": e.share_pct,
                "start_tick": e.start_tick, "end_tick": e.end_tick, **e.stats()}

    def _rollback_target(self):
        for _, version in reversed(self.deploy_history):
            if version != self.serving_version:
                return version
        return None

    def pipeline(self):
        with self.lock:
            sim = self.sim
            kafka = sim.get_service("Kafka")
            redis = self.system["redis"]
            datalake = self.system["datalake"]
            store = self.system["feature_store"]
            r = self.retrain
            recent = [self.tracker.counts.get(t, (0, 0))[0] for t in range(max(1, self.tick - 49), self.tick + 1)]
            events_per_tick = (sim.event_count / self.tick) if self.tick else 0
            return {
                "tick": self.tick,
                "kafka": {"events": sim.event_count, "events_per_tick": round(events_per_tick, 1),
                          "topics": {k: len(v) for k, v in kafka.topics.items()}},
                "flink": {"users_with_features": sum(1 for (f, _) in redis.store if f == "activity_level"),
                          "lag": kafka.lag("Flink"), "processed_last_tick": self.flink.processed_last_tick,
                          "capacity": self.flink.max_events_per_tick},
                "redis": {"keys": len(redis.store)},
                "exploration": self._exploration_view(),
                "server": {"model_version": self.serving_version, "slate_size": self.model_server.slate_size,
                           "num_candidates": self.model_server.num_candidates,
                           "slates_last_50_ticks": sum(recent),
                           "fallback_share": self.health["fallback_share"]},
                "ann": {"index_version": self.index_version, "model_version": self.serving_version,
                        "self_test": self.health["index_self_test"],
                        "items": len(getattr(self.retriever.ann_index, "nodes", {}) or {})},
                "lake": {**datalake.summary(), "lag": kafka.lag("DataLake")},
                "spark": self._spark_view(),
                "ofs": {"users": len(store.user_features), "items": len(store.item_features)},
                "train": None if r is None else {"version": r["version"], "started_tick": r["started_tick"],
                                                 "deploy_tick": r["deploy_tick"], "done": r["done"].is_set(),
                                                 "stage": r["stage"]},
                "workflow": self._workflow_view(),
                "registry": {"versions": len(self.versions), "serving_version": self.serving_version},
                "redis_skew": self.health["feature_skew"],
                "world": {"kind": self.world_kind, **self.dynamics.summary()},
                "infra": {
                    "kafka": kafka.describe(),
                    "flink": self.flink.describe(sim),
                    "redis": redis.describe(),
                    "feast": self.feast.describe(self.tick),
                    "kserve": self.kserve.describe(),
                    "ann_index": self._index_view(),
                },
                "money": self.economy.view(),
                "swappable": {name: {kind: {"name": v["name"], "about": v["about"],
                                            "price_per_tick": v["price_per_tick"]}
                                     for kind, v in options.items()}
                              for name, options in self.SWAPPABLE.items()},
                "settings": dict(self.settings),
            }

    def _exploration_view(self):
        e = getattr(self, "slot_explorer", None)
        if e is None or not self.settings["explore_slots"]:
            return None
        model = self.flink.preferences
        active = self.system["world"].active_user_ids or list(self.system["world"].user_objects)
        clicks = {(ev.timestamp, ev.user_id): ev.movie_id for ev in self._recent_events(300)
                  if type(ev).__name__ == "RecommendationClickedEvent"}
        since = self.tick - 300
        explored = [k for k in list(e.explored)[-3000:] if k[0] > since]
        clicked = sum(1 for tick, user, movie in explored if clicks.get((tick, user)) == movie)
        return {**e.summary(), "drifting_users": sum(1 for u in active if model.drifting(u)) if model else 0,
                "explored_click_rate": clicked / len(explored) if explored else None}

    def _index_view(self):
        index = self.retriever.ann_index
        kind = index_kind(index)
        touched = vectors_per_query(index) if kind != "other" else None
        return {"kind": self.index_kind, "serving_kind": kind,
                "name": INDEX_TYPES.get(kind, {}).get("name", kind),
                "items": len(getattr(index, "nodes", {}) or {}) if kind != "other" else None,
                "vectors_per_query": touched,
                # Simulated search time: 1 microsecond per vector touched.
                "simulated_search_ms": None if touched is None else round(touched / 1000.0, 2),
                "nprobe": self.settings["ann_nprobe"], "self_test": self.health["index_self_test"],
                "swap_cost": ONE_OFF["swap_ann_index"]}

    def incidents_view(self):
        with self.lock:
            return [dict(i, timeline=[{"tick": t, "text": x} for t, x in i["timeline"]])
                    for i in reversed(self.incidents)]

    def faults_view(self):
        """Faults so far. Hidden chaos faults are only counted while active
        (the player has to find them); drills and ended faults are shown."""
        with self.lock:
            shown, hidden_active = [], 0
            for fault in self.faults.faults:
                drill = fault.id in self.drills
                if fault.status == "active" and not drill:
                    hidden_active += 1
                    continue
                if fault.status == "scheduled" and not drill:
                    continue
                shown.append({**fault.summary(), "drill": drill})
            return {"hidden_active": hidden_active, "faults": list(reversed(shown)),
                    "kinds": FAULT_KINDS, "fixes": {k: list(v) for k, v in FIXES.items()}}

    def insight(self):
        """Stage-by-stage report (console/insight.py). Uses ground truth;
        read-only and deterministic (no RNG). Computed on request."""
        from console.insight import stage_report

        with self.lock:
            self.last_insight = stage_report(self)
            return self.last_insight

    def actions_pending(self):
        with self.lock:
            return list(self._pending)


def ctr_drop_alert(baseline, current, drop_pct):
    """True when `current` CTR is more than drop_pct % below `baseline`."""
    if baseline is None or current is None or baseline <= 0:
        return False
    return current < baseline * (1 - drop_pct / 100.0)



def _pct(value):
    return "–" if value is None else "%.1f%%" % (value * 100)


def recall_at_k(user_tower, index, holdout, k=20):
    """Share of held-out clicks whose movie is in the model's top k."""
    if not holdout:
        return None
    hits = sum(movie_id in index.search(user_tower.embed(features), k) for features, movie_id in holdout)
    return hits / len(holdout)
