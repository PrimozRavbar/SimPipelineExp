# SimPipeline User Guide

This guide covers everything you can see and do in the SimPipeline console: the screens, every
component and its settings, money, faults and repairs, and strategies that work. The
[README](../README.md) has the short version and the install steps.

- [Starting a game](#starting-a-game)
- [How the game works](#how-the-game-works)
- [The screens](#the-screens)
- [The components and their settings](#the-components-and-their-settings)
- [Models: retraining, deploying, rolling back](#models-retraining-deploying-rolling-back)
- [Money and compute credits](#money-and-compute-credits)
- [Faults, incidents and repairs](#faults-incidents-and-repairs)
- [The world: static or dynamic users](#the-world-static-or-dynamic-users)
- [Strategies that work](#strategies-that-work)
- [Glossary](#glossary)
- [FAQ](#faq)

---

## Starting a game

```bash
pip install -e ".[console]"
```

```bash
python -m console --taste rich
```

Then open <http://localhost:8000>. Options:

| Option | Default | Meaning |
|---|---|---|
| `--taste` | `genre` | `rich`: users have hidden tastes and movies have a quality beyond their genres (recommended). `genre`: taste is genres only (easier). `personal`: hidden personal tastes count three times as much (hardest; more to gain). |
| `--world` | `static` | `dynamic`: tastes drift, users tire of repeats, new users sign up over time. |
| `--preset` | `best` | `best`: the strongest measured setup. `plain`: every model feature off, so you build the stack up yourself. |
| `--seed` | `42` | Same seed and same actions at the same ticks give the same game. |
| `--users-per-tick` | `20` | Traffic. More users learn faster but run slower. |
| `--port` | `8000` | Web port. |

On Colab, open `SimPipExp_console.ipynb`, set `WORLD`, `TASTE` and `PRESET` in the first cell and
run every cell. The console opens in a new browser tab. To stop the server, interrupt the
server cell (locally: Ctrl+C in the terminal).

## How the game works

The simulation advances in **ticks**. Each tick, 20 users (by default) visit, each gets a slate
of 10 movies from your recommender, and each may click one, watch it and rate it. Those
events flow through your pipeline: they update the online features the next request uses and
pile up in the data lake for the next retrain.

**Click-through rate (CTR)** is the share of slates that got a click. It is the headline
number. Two reference lines frame it: a **random** slate (about 12%) and the **perfect** slate
(about 69%), the best ten movies for each user if you knew their true tastes. You never do: the
game knows them, your models have to learn them from clicks.

**Actions apply at the next tick boundary**, in the order you made them. If the clock is
paused, press play or step to apply them. This is what makes the game deterministic: same seed,
same actions at the same ticks, same game.

The clock runs at 1× (2 ticks a second), 10× or 100× (as fast as your machine allows).

## The screens

### Top bar

Clock and speed controls, the **production model** (and whether the ANN index matches it), the
**compute budget** (credits for training and Spark) and your **cash** with the profit or loss
of the last tick. The badge shows the seed, the taste model and the preset.

### Missions *(preview)*

Shows the free-play objectives. Scripted missions with scoring are coming; this screen is a
preview and marked as mock.

### Pipeline

The whole system as a diagram in three lanes:

- **Serving path** (every request): Kafka, Flink, Redis, Feast, KServe, the model server, the
  ANN index.
- **Training path** (offline): data lake, Spark, offline features, training, model registry.
- **World and chaos**: your users and fault injection.

Each node shows a live status: green is healthy, amber needs attention, red is broken. Select a
node to see what it does, what it is in a real company, its settings and its actions. Change
numbers and press **Apply at next tick** (or Enter); switches apply when you flip them. A
pending change is marked until it applies.

At the bottom, **Costs** shows what each component cost last tick, your revenue, and every
one-off charge.

### Models

Every model version with its training data, epochs, online CTR, ranker details and stage
(Production, Canary, Archived, Rejected). From here you **retrain**, **roll back** and
**export**. **Where clicks are lost** measures, using the users' true tastes, how good each
stage could be:

| Line | Meaning |
|---|---|
| Best possible (oracle) | the 10 most relevant movies in the whole catalog |
| Best of all candidates | the best 10 among the movies retrieval handed to the ranker. **Oracle minus this = what retrieval loses.** |
| Best of two-tower / extra candidates | the same, per candidate source |
| Candidates ordered by the learned profile | what you'd get by trusting only the system's genre profile of the user |
| Served | what users actually see. **Candidates minus this = what ranking loses.** |

### Monitoring

CTR over time with retrains and deploys marked, active users, request latency (real wall-clock
time on your machine), feature freshness, and health tiles: Flink lag, fallback share, ANN
index self-test, online/offline feature skew, catalog coverage and data drift. Pick the time
range at the top right.

### Incidents

Alerts opened by the monitoring rules, newest first. Each has a severity, a description, a
**runbook** (the first steps an on-call engineer would take), a timeline and the CTR around it.
Incidents resolve when their condition clears; CTR incidents are mitigated by a deploy or a
rollback. Below: every fault so far (hidden chaos faults are revealed when they end).

## The components and their settings

All settings live on the Pipeline screen. Defaults reproduce the standard game.

### Event bus (Kafka)

Every slate shown, click, watch and rating is published here. Flink, the data lake and the
metrics all read from it.

| Setting | Default | Effect |
|---|---|---|
| Partitions | 3 | Messages are keyed by user. Changing it is a repartition ($50). |
| Replication factor | 1 | Copies of each partition. With 1, a dead broker takes its partitions offline. |
| Brokers | 3 | $0.50 per broker per tick. |
| Retention (ticks) | 0 (forever) | Older messages are deleted. A consumer that falls further behind loses them. |

Action: **Restart failed broker**.

### Stream features (Flink)

Reads events from Kafka and keeps the online features up to date: user taste profiles, recent
watches, movie signals, the preference model.

| Setting | Default | Effect |
|---|---|---|
| Parallelism (task slots) | 1 | $0.80 per slot per tick. Changing it is a savepoint and restart: 2 ticks without processing, $20. |
| Events per slot per tick | 0 (unlimited) | Capacity = slots × this. Below the event rate, lag builds up. |
| Hard cap | 0 (none) | Caps the whole job. |
| Lag alert at | 150 | Opens an incident when Flink is this far behind. |

The node shows **backpressure** when lag keeps growing. Action: **Restart Flink**.

### Online features (Redis)

The fast store serving reads from.

| Setting | Default | Effect |
|---|---|---|
| Memory limit (keys) | 0 (unlimited) | $0.10 per 1,000 keys per tick. Over the limit, keys are evicted. |
| Eviction policy | allkeys-lru | `allkeys-lru` evicts the least recently used key, `volatile-ttl` the one closest to expiring, `noeviction` refuses new writes. |
| Feature TTL (ticks) | 0 (never) | User features expire this long after their last update; serving then treats the user as new. |
| Rich taste profiles | on (best preset) | Profiles from clicks, watch completion and ratings, recent activity first (off: watches only). |
| Skew alert at (%) | 60 | Incident when online and offline profiles disagree this much. |

### Feature store (Feast)

Every feature the serving path reads goes through Feast. Stream features come from Flink via
Redis; batch features (Spark) can be served two ways:

| Setting | Default | Effect |
|---|---|---|
| Batch features served | direct | `direct`: from the offline store on every request. `materialized`: from an online copy refreshed on a schedule, as most real deployments do; stale in between. |
| Materialize every (ticks) | 100 | $2 per run. 0 = never (the copy goes stale). |

### Inference service (KServe)

Runs the model server on replicas.

| Setting | Default | Effect |
|---|---|---|
| Min / max replicas | 1 / 1 | $1 per replica per tick. The autoscaler moves between them. |
| Requests per replica per tick | 0 (unlimited) | With a limit, requests beyond the ready replicas get the popularity fallback (overflow). |
| Target utilisation (%) | 70 | Lower = more headroom for spikes, more replicas. |
| Cold start (ticks) | 3 | Time for a new replica to be ready. |

The node shows replicas and estimated latency (which rises with utilisation). More than 10%
overflow opens an incident.

### Model server

Builds each slate: retrieve candidates, rank them, re-rank, show the top.

| Setting | Default (best) | Effect |
|---|---|---|
| Retrieved candidates | 100 | Candidates handed to the ranker. |
| Slate size | 10 | Movies the user sees. |
| Profile / co-watch / preference candidates | 30 / 30 / 50 | Extra candidate sources merged with the two-tower ones (0 = off). |
| Diversity (%) | 15 | MMR on genres: some variety helps when users tire of repeats; too much hurts. |
| Exploration (%) | 0 | Share of slots given to random lower candidates: fewer clicks now, more data. |
| Preference model | on | Learns from every logged signal, online every tick; ranks the candidates. |
| Learn repeat effects | on | The preference model also sees how often a movie was shown and whether it was watched. |
| Ranking stage, MLP ranker, per-genre features, movie signals, retrieval score | on | The learned ranker and its inputs (trained at the next retrain). |
| Multi-task ranker | off | Also predicts watch completion. |
| Filter watched movies | off | Never show a movie the user already watched. |
| Movie / genre exploration bonus | 0 / 0 | The preference model tries movies rarely shown to anyone / genres this user has rarely seen. Costs clicks now; can find tastes that moved. |
| Taste memory (ticks) | 0 | Each user's taste follows their recent behaviour; older evidence halves every N ticks. |
| Collaborative filtering, CF candidates, CF factors | off, 0, 16 | "People with clicks like yours liked these": hidden taste numbers learned from everyone's clicks at each retrain, used by the preference model and as a candidate source. Needs a lot of clicks per user; helps a little in the personal world. |
| Exploration slots / rate / rate for drifting users | 1 / 10% / 50% in the dynamic world (off in the static one) | Targeted exploration: a slot shows a well-rated movie from a genre the model is unsure about for this user, from the whole catalog. Users whose clicks dropped or whose browsing moved explore more. |
| Which genre to explore | thompson | `thompson` samples from what the model believes and how unsure it is; `uniform` favours genres the user has seen least. |
| Learn from exploration (IPS) | on | Explored picks are weighted by 1 / the probability they were shown, so the model learns what users think of them. |

Action: **Restart model server**.

### ANN index (swappable)

Holds one vector per movie; the user's vector finds its nearest movies here. It must come from
the same model version as the user tower, or users and movies stop being comparable.

**Swap** it for another type ($500 migration; rebuilt at the next tick; new models use it):

| Type | Recall | Search cost | Price per tick |
|---|---|---|---|
| HNSW (graph) | ~96% | ~370 vectors per query | $0.60 |
| IVF (FAISS-style) | depends on `nprobe` | ~380 at nprobe 4 | $0.30 |
| Exact (brute force) | 100% | every vector (1,682) | $1.50 |

Settings: **IVF cells searched (nprobe)**; **Rebuild index on deploy** (off: a deploy keeps the
old index, the classic stale-index bug). Action: **Rebuild index**.

### Data lake, Spark, offline features

The data lake keeps every event; its sink writes one file per topic per tick, and a compaction
job merges the small files (`Compact small files every`, default 100 ticks). The offline feature
store keeps point-in-time snapshots, so training sees each feature as it was when the event
happened.

**Spark** recomputes offline features on a schedule (`Run every`, default 100 ticks, about once
a day) and as the first stage of every retrain. A job runs on a simulated cluster: it reads the
lake files, shuffles every event to its user's and its movie's partition, computes features
task by task on the executors and commits a new feature version at once. A failed job publishes
nothing: the store keeps the last good version. **One tick is about 15 minutes**: a healthy job
takes a minute or two and is ready at once; a slow one finishes ticks later (and a retrain waits
for it). The features are the same whatever the cluster: only the time and the failures change.

| Setting | Default | Effect |
|---|---|---|
| Executors, cores per executor | 4, 4 | Task slots. Fewer = slower jobs; you pay per executor-hour while a job runs. |
| Memory per executor (GB) | 8 | A task's share is memory x 0.6 / cores. More rows than that spill to disk (slower); far more and the container is killed (out of memory). |
| Shuffle partitions | 200 | Reduce tasks per stage. Too few: big tasks that spill. Too many: overhead (AQE merges them). |
| Max task failures | 4 | Attempts per task before the job is aborted. |
| Incremental runs | on | Only the events since the last run. Changing the profile settings triggers a full backfill. Off: everything, every run (slow, grows with the lake). |
| Adaptive query execution | on | Merges small partitions, splits skewed ones (a hot movie) into several tasks. A single movie is never split. |
| Speculative execution | on | A straggling task gets a copy on another executor; the first to finish wins. |
| Wait for complete data | on | The job waits until the lake has every tick so far, and fails after `Wait for data at most` (8 ticks). Off: it runs on whatever has landed and "succeeds" on stale data. |

The Spark panel (Pipeline > Batch features) shows the last job: its stages (tasks, rows, median
and slowest task, skew, spill, failed attempts), the executors over time (lost executors in
red), the job log and the last 20 jobs. Actions: **Run feature job now** (5 credits),
**Replace degraded node**; on the data lake, **Restart lake sink**.

### Training

| Setting | Default | Effect |
|---|---|---|
| Retrain every (ticks) | 0 | Automatic retrains (40 credits each). 0 = only when you start one. |
| Epochs, negatives | 5, 5 | Training length and sampled negatives per click. |
| Deploy after (ticks) | 50 | The release window: the earliest a new model serves after its retrain starts. A training job that takes longer deploys when it is done. |
| Popularity correction (logQ) | on | Stops popular movies being pushed down by in-batch negatives. |
| Graded positives, random negatives, recent history, engagement, demographics, shared IDs, train from scratch | off | Advanced training options. |
| Batched training | on | Much faster on CPU; same model. |
| GPUs | 1 | Data-parallel training: faster, $3 per GPU-hour, a little lost to communication. |
| GPU memory (GB) | 16 | The weights, the batch's activations and the batch x batch in-batch logits must fit, or the job dies with CUDA out of memory. |
| Batch size | 32 | Real: bigger batches keep the GPU busy (faster) and add in-batch negatives, but take fewer steps per epoch and need more memory. |
| Learning rate | 0.001 | Real: too high or too low and the model trains worse. |
| Checkpoint every (minutes) | 10 | A preempted job resumes from its last checkpoint; 0 = it restarts from scratch. |
| Validate training data | on | Stop the run if the training set looks broken (see below). |
| Validate the model | on | Do not deploy a model whose loss is broken or that is worse than the serving model on the same new data. |

**Every retrain is a workflow** (an Airflow DAG run), shown step by step under Pipeline >
Training: **features** (the Spark job), **validate data**, **train**, **validate model**,
**evaluate** (with the gate on) and **deploy**. A failed step stops the run and the serving
model stays; an incident opens with the failing step.

- **Data validation** compares the training set with the last one that passed: at least 60% as
  many examples, a click rate at least half the last one's, and no more than 15 points more
  users with an empty taste profile among the clicks since the last run (a broken feature job
  shows up there first).
- **Training** runs on a simulated GPU cluster (one tick is about 15 minutes): start-up, loading
  the data, the epochs (examples x candidates / GPU throughput), the index and the rankers, saving
  the model. A healthy run takes minutes and deploys on the release window; a slow one (GPUs
  scarce, many preemptions) deploys late. The real training runs in the background either way:
  the GPU settings change when the model is ready, never the model (batch size and learning
  rate do change the model).
- **Model validation** checks that the loss is a number, did not go up during training and is
  no more than 10% above the serving model's on the same new examples (a sample of the training
  set), and that the movie embeddings are finite and not all the
  same.

### Model registry

| Setting | Default | Effect |
|---|---|---|
| Alert: CTR drop / within / window | 20% / 100 / 50 | The deploy alert rule. |
| Canary share (%) / length | 0 / 150 | Serve a new model to a share of users first; it is promoted unless it loses with 95% confidence. 50% = an A/B test. |
| Offline evaluation gate, held-out %, tolerance | off, 10, 10 | Reject a new model whose Recall@20 on held-out clicks is much worse than production's. |

## Models: retraining, deploying, rolling back

- **Retrain** (Models or Pipeline > Training): a workflow run. Spark builds features, the data is
  validated, then the two-tower model, the ranker and the preference model train in the
  background while the current model keeps serving. The model is validated and deploys
  `Deploy after` ticks after the start (later if training takes longer).
- **Roll back** redeploys the previous version instantly, including its index and its
  preference model. Some things are not versioned and keep learning whatever serves: user
  profiles and co-watch signals.
- **Canary**: with a canary share above 0, a new model first serves part of your users. Promote
  or abort it from Models, or let the test decide.
- **Export** writes a model bundle to the run directory.

The **compute budget** (1,000 credits) limits how often you can retrain (40) and run Spark (5).

## Money and compute credits

Two resources:

- **Compute credits** pay for training runs and Spark jobs.
- **Cash** pays for the infrastructure. You start with $10,000. Every click earns $1. Every tick
  you pay for your configuration:

| Item | Price |
|---|---|
| Kafka | $0.50 per broker + $0.05 per partition, per tick |
| Flink | $0.80 per task slot per tick |
| Redis | $0.20 + $0.10 per 1,000 keys per tick |
| KServe | $1.00 per replica per tick |
| ANN index | HNSW $0.60, IVF $0.30, exact $1.50 per tick |
| Feast | $0.30 per tick + $2 per materialization run |
| Training | $3 per GPU-hour while a training job runs (a default retrain costs about $1-3) |
| Spark | per executor-hour while a job runs: $0.50 per core + $0.10 per GB (the default cluster $11.20 an hour; a healthy job costs well under a dollar) |
| One-off | index swap $500, Flink rescale $20, Kafka repartition $50 |

The default setup costs about $4.50 a tick. At 20 users a tick an untrained model (CTR ~13%)
loses money; a trained one (CTR 30-50%) makes it. Going below $0 opens an incident.

## Faults, incidents and repairs

Inject a known fault from **Pipeline > Chaos** (a drill, 100 ticks), or set `Random fault about
every` to get hidden ones.

| Fault | Symptoms | Fix |
|---|---|---|
| Flink stream job down | lag grows, features go stale, CTR sags | Restart Flink |
| Online user features corrupted | CTR barely moves, feature skew jumps | Restart Flink |
| ANN index returns wrong movie ids | CTR drops, index self-test fails | Rebuild index (or a deploy) |
| Two-tower retrieval down | fallback share 100% | Restart model server |
| Traffic spike | more users per tick; overflow if KServe is capped | Scale KServe, wait |
| Spark feature jobs failing (cluster down) | jobs fail, features age, retrains abort | Wait; freshness alert |
| Spark executors being reclaimed (spot) | executors lost mid-job (each lives ~10 minutes), tasks retried, lost shuffle output recomputed; slower jobs | The default cluster absorbs it; a small cluster gives up after max(2 x executors, 3) lost executors. More executors, incremental runs |
| Degraded Spark node | one executor 10x slower: stragglers, high skew ratio | Speculation (on by default); Replace degraded node |
| Spark cluster busy | jobs wait ~30 min for resources and get half the executors; late features | Wait |
| Data lake sink stuck | lake stops growing, its Kafka lag grows; jobs wait for data, then fail | Restart lake sink |
| A movie went viral | one movie everyone clicks and watches (World): a hot key in every job, a shift in what users want | Retrain; AQE and memory for the jobs |
| Training GPUs being preempted (spot) | training jobs restart from their last checkpoint and take longer; without checkpoints they start over and can fail | Keep checkpoints on; wait |
| Feature job writing empty profiles | a bad deploy of the Spark job: most users' profiles go empty in the offline store, and stay so after the fault ends; the next retrain fails data validation | Backfill features (Pipeline > Batch features), then retrain |
| Workflow scheduler down | scheduled Spark runs and retrains silently do not start; features and models go stale | Restart workflow scheduler (Pipeline > Training) |
| Kafka broker 0 down | partitions offline (with replication 1), lag builds | Restart failed broker; use replication 2 |

Incident rules: CTR drop (any time and after a deploy), Flink lag, fallback share, ANN
self-test, feature skew, freshness, data drift, cash below zero, Kafka partitions offline,
serving overflow, Spark job failed, Spark job over a tick, data lake behind, retrain workflow
failed and scheduled runs missed. Each comes with a runbook.

## The world: static or dynamic users

The users are MovieLens volunteers: their genre preferences and hidden tastes come from 100,000
real ratings. In the **rich** world (recommended) a movie's appeal is 40% genre fit and 60%
hidden taste and quality; genres alone get you only to ~40% CTR. A user scans a slate from the
top (lower positions are seen less), clicks at most one movie, and maybe watches and rates it.

The **dynamic** world adds:

| Setting | Default (dynamic) | Effect |
|---|---|---|
| Taste drift every / users drifting | 200 ticks / 5% | Some users' taste moves towards one genre. |
| How taste drifts | gradual | `gradual`: in small steps towards a genre related to what they like. `abrupt`: most of the taste jumps to a random genre at once (harder). |
| Fatigue (%) | 15 | Each repeat impression of a movie lowers its click chance by 15%. |
| Fatigue wears off (ticks) | 1000 | Fatigue halves every 1000 ticks since the last impression. |
| Rewatch (%) | 10 | A watched movie keeps 10% of its appeal. |
| New users per 100 ticks | 30 | 30% of users join later, with no history (cold start). |
| Onboarding | on (best) | New users name 3 favourite genres (imperfectly) before their first slate. |
| Saves / not interested (%) | 100 / 50 (both worlds) | Users who look at a movie they'd like but don't click sometimes save it (more often the better it fits); movies that don't fit them sometimes get "not interested". The strongest signal the system gets besides clicks. |
| Ratings (% of watches) | 10 | How often a user rates a movie they watched. |
| Slates per visit | 1 | Rows a user scrolls per visit: more clicks per visit, but each row is second-best and users tire faster. |
| Browsing (% of visits) | 30 (both worlds) | Users sometimes browse or search a genre before their slate: explicit evidence of their current taste, which the profiles and the preference model learn from. |
| A movie goes viral every / for (ticks) | 0 (off) / 300 | Everyone talks about one random movie: its click chance moves halfway to the maximum for every user, and 15% of visiting users watch it without a recommendation. A hot key for every batch job. Also a drill and a chaos fault. |

## Strategies that work

- **Retrain early, then regularly.** The first retrain is the biggest jump; later ones keep
  learning as users build history. Set `Retrain every` to 400 when you're comfortable.
- **Read *Where clicks are lost* before changing anything.** If retrieval loses most, add or
  tune candidate sources; if ranking loses most, look at the ranker.
- **Variety pays when users get bored.** In the dynamic world, a little diversity helps; too
  much always hurts. In the static world, leave it low.
- **Replicate what you can't afford to lose.** Kafka replication 2 survives a broker outage for
  $1.50 more a tick.
- **Leave headroom.** A capped KServe at 100% target utilisation drops requests to the fallback
  during a traffic spike before the autoscaler catches up.
- **Swap the index when it's worth it.** IVF halves the index bill; check the self-test and
  CTR after the swap and raise `nprobe` if recall suffers.
- **Use canaries for risky changes.** A 10% canary for 150 ticks catches a bad model before
  everyone sees it.
- **Watch the money.** Every component you add has a running cost; every click pays $1.

## Glossary

| Term | Meaning |
|---|---|
| Tick | One step of simulated time; each tick, a batch of users visit. |
| Slate | The 10 movies a user is shown. |
| CTR | Click-through rate: the share of slates that got a click. |
| Oracle | The best possible slate, computed from the users' true tastes. |
| Candidates | The movies retrieval hands to the ranker (100 by default). |
| Two-tower model | A user network and a movie network trained so a user's vector lands near movies they'll click. |
| ANN index | Approximate nearest-neighbour search over movie vectors. |
| Ranker / re-ranker | Orders the candidates / then adjusts the order (diversity, exploration, filters). |
| Train/serve skew | Features at serving time that differ from what the model saw in training. |
| Point in time | Training sees each feature as it was when the event happened, never later. |
| Canary | A new model serving a small share of users before everyone. |
| Consumer lag | Messages published to Kafka but not yet processed by a consumer. |
| Materialization | Copying batch features into the online store. |

## FAQ

**Why did CTR barely move after my retrain?** The first models learn mostly from what the
untrained model showed; give it data and retrain again. Check that the index matches the model
(top bar) and look at *Where clicks are lost*.

**Why does CTR drop in the dynamic world over long games?** Users tire of movies they've seen
repeatedly and can't rewatch, so the system must keep finding fresh movies, and some users'
tastes drift towards new genres. Browsing and targeted exploration (on by default in the dynamic
world) keep CTR flat over long games; reaching the oracle there needs much finer knowledge of each
user than a few clicks give. It's the dynamic world's open challenge.

**Why is latency different on my machine?** Latency is the real time your computer takes to
build a slate; it never affects the simulation.

**Can I replay a game?** Yes: same seed, same actions at the same ticks.

**Where are the logs?** Each game writes `runs/<run id>/metrics.jsonl`;
`dashboard.render_dashboard(run_dir)` turns it into a one-page report.
