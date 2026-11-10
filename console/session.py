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
import statistics
import threading
import time

from ml_system.ann import HNSWIndex
from ml_system.build_recommendation_system import build_recommendation_system
from ml_system.metrics import reference_ctrs
from world.external_data_loader import load_movielens


# Ticks per second at speed 1x (the mockup advanced 1 tick every 500 ms).
BASE_TICKS_PER_SECOND = 2
SPEEDS = (1, 10, 100)
MIN_YIELD_SECONDS = 0.003

# Compute budget: a simple game resource, not a measurement.
BUDGET_TOTAL = 1000
COST_RETRAIN = 40
COST_SPARK = 5

DEFAULT_SETTINGS = {
    # Offline schedules (0 = off, the player triggers runs by hand).
    "spark_every": 0,          # run the Spark feature job every N ticks
    "retrain_every": 0,        # start a retrain every N ticks
    "deploy_delay": 50,        # ticks between retrain start and deploy
    # Training
    "epochs": 5,
    "negatives": 5,            # sampled negatives per positive
    # Serving
    "slate_size": 10,
    "num_candidates": 100,
    "rebuild_index": True,     # rebuild the ANN index with the new item tower on deploy
    # Alert rule: open an incident when rolling CTR drops more than
    # alert_drop_pct % within alert_ticks ticks after a deploy.
    "alert_drop_pct": 20,
    "alert_ticks": 100,
    "alert_window": 50,
}

# (min, max) for numeric settings; bools are listed in BOOL_SETTINGS.
SETTING_LIMITS = {
    "spark_every": (0, 100000),
    "retrain_every": (0, 100000),
    "deploy_delay": (1, 10000),
    "epochs": (1, 50),
    "negatives": (1, 9),
    "slate_size": (1, 100),
    "num_candidates": (1, 500),
    "alert_drop_pct": (1, 100),
    "alert_ticks": (10, 10000),
    "alert_window": (5, 1000),
}
BOOL_SETTINGS = {"rebuild_index"}

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

    def __init__(self, version, user_tower, item_tower, index, **info):
        self.version = version
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
        users_per_tick=10,
        settings=None,
        data=None,
        log_metrics=True,
        runs_dir="runs",
        run_id=None,
        training_wait_timeout=None,
    ):
        self.seed = seed
        self.users_per_tick = users_per_tick
        self.settings = dict(DEFAULT_SETTINGS)
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
        )

        self.sim = self.system["sim"]
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
        self.tick_versions = {}          # tick -> model version that served it

        # Offline jobs
        self.last_spark_tick = None
        self.spark_runs = 0

        # Budget, event log, incidents
        self.budget_used = 0
        self.log = []
        self.incidents = []
        self._watch = None               # deploy being watched by the alert rule

        # Serving latency: wall-clock ms of ModelServer.recommend per request.
        self.latency = {}                # tick -> list of ms
        self._wrap_recommend()

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
        self.sim.tick()
        self._check_alert()

    def _boundary(self):
        now = self.tick

        # 1. A due deploy goes first, so actions at this boundary see it.
        if self.retrain is not None and now >= self.retrain["deploy_tick"]:
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
        if now > 0 and s["spark_every"] and now % s["spark_every"] == 0:
            self._run_spark(by="schedule")
        if now > 0 and s["retrain_every"] and now % s["retrain_every"] == 0:
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

    ACTIONS = ("retrain", "rollback", "spark", "settings")

    def submit(self, name, **params):
        """Queue an action for the next tick boundary. Returns its record;
        `status` becomes "applied" or "rejected" once the boundary runs."""
        if name not in self.ACTIONS:
            raise ActionError("unknown action %r" % name)
        if name == "settings":
            self._check_settings(params)
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
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise ActionError("%s must be a whole number" % key)
            low, high = SETTING_LIMITS[key]
            if not low <= value <= high:
                raise ActionError("%s must be between %d and %d" % (key, low, high))

        slate = values.get("slate_size", self.settings["slate_size"] if hasattr(self, "settings") else 10)
        cands = values.get("num_candidates", self.settings["num_candidates"] if hasattr(self, "settings") else 100)
        if slate > cands:
            raise ActionError("slate size cannot exceed retrieved candidates")

    def _apply_settings(self, values):
        self._check_settings(values)
        changed = {k: v for k, v in values.items() if self.settings[k] != v}
        for key, value in changed.items():
            self.settings[key] = int(value) if key not in BOOL_SETTINGS else value
        self.model_server.slate_size = self.settings["slate_size"]
        self.model_server.num_candidates = self.settings["num_candidates"]
        if changed:
            text = ", ".join("%s=%s" % (k, v) for k, v in sorted(changed.items()))
            self._log("config", "Settings changed: %s." % text, cls="s-accent")
            if self.logger is not None:
                self.logger.log_event("settings", self.tick, **changed)
        return "settings applied"

    # Spark -------------------------------------------------------------

    def _run_spark(self, by):
        events = self.system["datalake"].events
        self.system["spark"].run_offline_feature_job(events, self.offline.feature_pipeline)
        self.last_spark_tick = self.tick
        self.spark_runs += 1
        self.budget_used += COST_SPARK
        self._log("job", "Spark feature job ran over %d events (by %s)." % (len(events), by))
        if self.logger is not None:
            self.logger.log_event("spark", self.tick, events=len(events), by=by)
        return "spark job finished"

    # Retrain / deploy --------------------------------------------------

    def _start_retrain(self, by):
        if self.retrain is not None:
            raise ActionError("a training run is already in progress (deploys at tick %d)" % self.retrain["deploy_tick"])
        if self.budget_used + COST_RETRAIN > BUDGET_TOTAL:
            raise ActionError("not enough compute budget left for a retrain")

        now = self.tick
        version = max(self.versions) + 1
        events = list(self.system["datalake"].events)

        # Everything that reads shared state or uses the session RNGs runs
        # here, on the clock thread at a tick boundary: offline features,
        # the training set (negative sampling, shuffle) and the item
        # features the new item tower will embed.
        self.offline.feature_pipeline.process(events)
        generator = self.offline.dataset_generator
        generator.negative_samples = self.settings["negatives"]
        for order, event in enumerate(events):
            generator.add_event(event, order)
        training_data = generator.build()
        if not training_data:
            raise ActionError("no clicks to learn from yet; let the simulation run first")
        items = _ItemFeatureSnapshot(self.offline.feature_pipeline.feature_store)

        self.budget_used += COST_RETRAIN
        deploy_tick = now + self.settings["deploy_delay"]
        record = {
            "version": version,
            "started_tick": now,
            "deploy_tick": deploy_tick,
            "training_examples": len(training_data),
            "epochs": self.settings["epochs"],
            "negatives": self.settings["negatives"],
            "rebuild_index": self.settings["rebuild_index"],
            "by": by,
            "done": threading.Event(),
            "result": None,
            "error": None,
            "thread": None,
        }
        thread = threading.Thread(
            target=self._train, args=(record, training_data, items),
            name="retrain-v%d" % version, daemon=True,
        )
        record["thread"] = thread
        self.retrain = record
        thread.start()

        self._log("train", "Training v%d started on %d examples (%d epochs, %d negatives, by %s). Deploys at tick %d."
                  % (version, len(training_data), record["epochs"], record["negatives"], by, deploy_tick), cls="s-accent")
        if self.logger is not None:
            self.logger.log_event("retrain_start", now, model_version=version,
                                  training_examples=len(training_data), deploy_tick=deploy_tick)
        return "training v%d, deploys at tick %d" % (version, deploy_tick)

    def _train(self, record, training_data, items):
        """Background thread: train, embed every movie, build the index.
        Touches only the trainer (never served directly) and its inputs."""
        try:
            started = time.monotonic()
            trainer = self.offline.trainer
            trainer.epochs = record["epochs"]
            trainer.train(training_data)

            user_tower = copy.deepcopy(trainer.user_tower).eval()
            item_tower = copy.deepcopy(trainer.item_tower).eval()

            embeddings = {
                movie_id: item_tower.embed(features)
                for movie_id, features in items.item_features.items()
            }
            index = HNSWIndex(seed=self.seed * 1000 + record["version"])
            index.build(embeddings)

            record["result"] = (user_tower, item_tower, index)
            record["train_seconds"] = round(time.monotonic() - started, 2)
        except Exception as error:  # surfaced in the log at deploy time
            record["error"] = repr(error)
        finally:
            record["done"].set()

    def _finish_retrain(self):
        record = self.retrain
        record["done"].wait(self.training_wait_timeout)
        record["thread"].join()
        self.retrain = None
        now = self.tick
        version = record["version"]

        if record["error"] is not None:
            self._log("train", "Training v%d failed: %s" % (version, record["error"]), cls="s-crit")
            return

        user_tower, item_tower, index = record["result"]
        self.versions[version] = ModelVersion(
            version, user_tower, item_tower, index,
            started_tick=record["started_tick"],
            trained_tick=record["started_tick"],
            deployed_tick=None,
            training_examples=record["training_examples"],
            epochs=record["epochs"],
            negatives=record["negatives"],
            train_seconds=record.get("train_seconds"),
            recall_at_20=None,
            note="",
        )
        if self.logger is not None:
            self.logger.log_event("retrain", now, model_version=version,
                                  training_examples=record["training_examples"])
        self._log("train", "Training v%d finished (%d examples, %.1f s wall time)."
                  % (version, record["training_examples"], record.get("train_seconds") or 0), cls="s-accent")
        self._deploy(version, by="auto" if record["by"] == "schedule" else record["by"],
                     rebuild_index=record["rebuild_index"])

    def _deploy(self, version, by, rebuild_index=True, note=""):
        model = self.versions[version]
        self.retriever.user_tower = model.user_tower
        self.retriever.item_tower = model.item_tower
        if rebuild_index:
            self.retriever.ann_index = model.index
            self.index_version = version
        previous = self.serving_version
        self.serving_version = version
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

        # Incidents opened after the previous deploy are mitigated by a new one.
        for incident in self.incidents:
            if incident["status"] == "Open":
                incident["status"] = "Mitigated"
                incident["timeline"].append((self.tick, "Model v%d deployed (by %s)." % (version, by)))

        self._watch_deploy(previous)

    def _rollback(self):
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
        self._log("alert", "%s opened: %s." % (incident["id"], incident["title"]), cls="s-crit")
        if self.logger is not None:
            self.logger.log_event("incident", self.tick, incident_id=incident["id"],
                                  model_version=watch["version"], baseline_ctr=watch["baseline"],
                                  current_ctr=current)

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
                "training": None if r is None else {
                    "version": r["version"],
                    "started_tick": r["started_tick"],
                    "deploy_tick": r["deploy_tick"],
                    "done": r["done"].is_set(),
                },
                "budget": {"used": self.budget_used, "total": BUDGET_TOTAL,
                           "retrain_cost": COST_RETRAIN, "spark_cost": COST_SPARK},
                "reference": self.reference,
                "run_dir": str(self.system["run_dir"]) if self.system["run_dir"] else None,
                "open_incidents": sum(1 for i in self.incidents if i["status"] == "Open"),
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
                    "stage": "Production" if version == self.serving_version else "Archived",
                })
            r = self.retrain
            training = None
            if r is not None:
                training = {k: r[k] for k in ("version", "started_tick", "deploy_tick",
                                              "training_examples", "epochs", "negatives", "by")}
                training["done"] = r["done"].is_set()
            return {
                "serving_version": self.serving_version,
                "index_version": self.index_version,
                "rollback_target": self._rollback_target(),
                "training": training,
                "versions": out,
            }

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
                "flink": {"users_with_features": sum(1 for (f, _) in redis.store if f == "activity_level")},
                "redis": {"keys": len(redis.store)},
                "server": {"model_version": self.serving_version, "slate_size": self.model_server.slate_size,
                           "num_candidates": self.model_server.num_candidates,
                           "slates_last_50_ticks": sum(recent)},
                "ann": {"index_version": self.index_version, "model_version": self.serving_version,
                        "items": len(getattr(self.retriever.ann_index, "nodes", {}) or {})},
                "lake": {"events": len(datalake.events)},
                "spark": {"last_run_tick": self.last_spark_tick, "runs": self.spark_runs,
                          "every": self.settings["spark_every"]},
                "ofs": {"users": len(store.user_features), "items": len(store.item_features)},
                "train": None if r is None else {"version": r["version"], "started_tick": r["started_tick"],
                                                 "deploy_tick": r["deploy_tick"], "done": r["done"].is_set()},
                "registry": {"versions": len(self.versions), "serving_version": self.serving_version},
                "settings": dict(self.settings),
            }

    def incidents_view(self):
        with self.lock:
            return [dict(i, timeline=[{"tick": t, "text": x} for t, x in i["timeline"]])
                    for i in reversed(self.incidents)]

    def actions_pending(self):
        with self.lock:
            return list(self._pending)


def ctr_drop_alert(baseline, current, drop_pct):
    """True when `current` CTR is more than drop_pct % below `baseline`."""
    if baseline is None or current is None or baseline <= 0:
        return False
    return current < baseline * (1 - drop_pct / 100.0)

