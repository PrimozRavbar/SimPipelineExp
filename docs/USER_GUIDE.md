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
| `--taste` | `genre` | `rich`: users have hidden tastes and movies have a quality beyond their genres (recommended). `genre`: taste is genres only (easier). |
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

The data lake keeps every event. **Spark** recomputes offline features on a schedule (`Run
every`, default 100 ticks) and as the first stage of every retrain; `Throughput` makes jobs take
time as the lake grows. The offline feature store keeps point-in-time snapshots, so training
sees each feature as it was when the event happened. Action: **Run feature job now** (5
credits).

### Training

| Setting | Default | Effect |
|---|---|---|
| Retrain every (ticks) | 0 | Automatic retrains (40 credits each). 0 = only when you start one. |
| Epochs, negatives | 5, 5 | Training length and sampled negatives per click. |
| Deploy after (ticks) | 50 | The new model serves this long after training starts (the clock waits if training is slower). |
| Popularity correction (logQ) | on | Stops popular movies being pushed down by in-batch negatives. |
| Graded positives, random negatives, recent history, engagement, demographics, shared IDs, train from scratch | off | Advanced training options. |
| Batched training | on | Much faster on CPU; same model. |

### Model registry

| Setting | Default | Effect |
|---|---|---|
| Alert: CTR drop / within / window | 20% / 100 / 50 | The deploy alert rule. |
| Canary share (%) / length | 0 / 150 | Serve a new model to a share of users first; it is promoted unless it loses with 95% confidence. 50% = an A/B test. |
| Offline evaluation gate, held-out %, tolerance | off, 10, 10 | Reject a new model whose Recall@20 on held-out clicks is much worse than production's. |

## Models: retraining, deploying, rolling back

- **Retrain** (Models or Pipeline > Training): Spark builds features, then the two-tower model,
  the ranker and the preference model train in the background while the current model keeps
  serving. The new version deploys `Deploy after` ticks later.
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
| Spark feature jobs failing | features age, retrains abort | Wait; freshness alert |
| Kafka broker 0 down | partitions offline (with replication 1), lag builds | Restart failed broker; use replication 2 |

Incident rules: CTR drop (any time and after a deploy), Flink lag, fallback share, ANN
self-test, feature skew, freshness, data drift, cash below zero, Kafka partitions offline and
serving overflow. Each comes with a runbook.

## The world: static or dynamic users

The users are MovieLens volunteers: their genre preferences and hidden tastes come from 100,000
real ratings. In the **rich** world (recommended) a movie's appeal is 40% genre fit and 60%
hidden taste and quality; genres alone get you only to ~40% CTR. A user scans a slate from the
top (lower positions are seen less), clicks at most one movie, and maybe watches and rates it.

The **dynamic** world adds:

| Setting | Default (dynamic) | Effect |
|---|---|---|
| Taste drift every / users drifting | 200 ticks / 5% | Some users' taste moves towards one genre. |
| Fatigue (%) | 15 | Each repeat impression of a movie lowers its click chance by 15%. |
| Fatigue wears off (ticks) | 1000 | Fatigue halves every 1000 ticks since the last impression. |
| Rewatch (%) | 10 | A watched movie keeps 10% of its appeal. |
| New users per 100 ticks | 30 | 30% of users join later, with no history (cold start). |
| Onboarding | on (best) | New users name 3 favourite genres (imperfectly) before their first slate. |

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
repeatedly and some tastes drift. Diversity and the preference model's repeat effects help.

**Why is latency different on my machine?** Latency is the real time your computer takes to
build a slate; it never affects the simulation.

**Can I replay a game?** Yes: same seed, same actions at the same ticks.

**Where are the logs?** Each game writes `runs/<run id>/metrics.jsonl`;
`dashboard.render_dashboard(run_dir)` turns it into a one-page report.
