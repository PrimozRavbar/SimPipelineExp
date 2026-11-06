# SimPipeline

**SimPipeline is a simulation environment for building and operating realistic machine-learning systems.**

The project is being developed toward a **gamified ML systems learning platform** where users build ML systems from interchangeable production components, operate them under realistic conditions, diagnose failures and performance problems, and see the effects of their decisions through system and business metrics.

The long-term goal is to simulate various ML systems. Thus SimPipeline is designed around reusable components that can support different ML systems, including recommendation, fraud detection, and other production ML workloads.

## Current State

Please run the DEMO notebook. You can simulate the world and the recommendation system ML pipeline receiving data from the world (clicks and views of the movies). The recommendations are also simulated at each time-step. This data will populate the datalake. Now you can run Spark to compute the offline features, and then train the Two-Tower model on them. Finally you can get the basic training quality metrics such as Recall@20. Now you have run the first simulation. After that you can keep running in for longer, re-training the model, etc, until the metric improves. Without the ranker and the re-ranker components implemented, a good result is about 60% Recall@20.

The current implementation is a **basic end-to-end recommendation system simulation**.

It already models the core production loop:

- simulated users interacting with recommendations
- recommendation and behavioral events
- Kafka-style event ingestion
- stream processing with Flink
- online state/features through Redis
- two-tower retrieval
- HNSW approximate nearest-neighbor retrieval
- model serving
- event storage in a DataLake
- periodic offline retraining
- Spark-based offline computation
- feature pipelines and feature storage
- training dataset generation
- two-tower model training
- embedding generation
- replacement of online retrieval artifacts after retraining

## Architecture

The current codebase of the system consists of an online serving loop and a periodic offline training loop.

                         ┌──────────────────────────┐
                         │        SIMULATOR         │
                         │      clock / runtime     │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │          WORLD           │
                         │      UserSimulator       │
                         └────────────┬─────────────┘
                                      │
                                      │ events
                                      ▼
                                   ┌───────┐
                                   │ Kafka │
                                   └───┬───┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                         ▼                           ▼
                     ┌───────┐                  ┌──────────┐
                     │ Flink │                  │ DataLake │
                     └───┬───┘                  └────┬─────┘
                         │                            │
                         │ online features            │ periodic
                         ▼                            ▼
                      ┌───────┐              ┌────────────────┐
                      │ Redis │              │ Spark          │
                      └───┬───┘              │ Offline       │
                          │                  │ Pipeline      │
                          │                  └───────┬────────┘
                          │                          │
                          │                          ├─ Feature Pipeline
                          │                          ├─ Training Dataset
                          │                          ├─ Two-Tower Training
                          │                          └─ Embedding Generation
                          │                          │
                          │                          ▼
                          │                    ┌──────────────┐
                          │                    │ HNSW Index   │
                          │                    └──────┬───────┘
                          │                           │
                          │       new towers + index  │
                          │              ┌────────────┘
                          │              │
                          ▼              ▼
                    ┌─────────────────────────┐
                    │   TwoTowerRetrieval     │
                    │                         │
                    │ User Tower              │
                    │ Item Tower              │
                    │ HNSW Index              │
                    └────────────┬────────────┘
                                 │
                                 │ recommendations
                                 ▼
                         ┌────────────────┐
                         │  ModelServer   │
                         └───────┬────────┘
                                 │
                                 │ recommendation
                                 │ events
                                 ▼
                              WORLD
                                 │
                                 └───────────────► next interaction

### The core loop

The central simulation loop is:

**World → Kafka → Flink → Redis → Retrieval → ModelServer → World**

Users interact with recommendations. Those interactions produce events. Events enter the streaming system, update online state/features, and become inputs to subsequent recommendation requests.

Recommendations themselves are represented as events, so the system forms a closed feedback loop rather than a one-way prediction pipeline.

### Retraining loop

Historical events are periodically used to retrain the retrieval system:

**Kafka → DataLake → Spark → Offline ML pipeline → new towers + HNSW → Retrieval**

The retraining process replaces the user tower, item tower, and ANN index used by the online retriever.

The current recommendation pipeline implements **candidate retrieval with a two-tower model**. The subsequent **ranking** and **re-ranking** stages are not yet implemented. They are planned components of the production-style recommendation architecture.

The current simulation also uses a **small MovieLens-based dataset**. This keeps the system manageable while the core architecture is being developed; increasing dataset scale and realism is part of the planned development.

## System Components

### Simulation

**Simulator**

Provides the simulation runtime, clock, service lifecycle, configuration, scenarios, and seeded random number generation.

**World**

Represents the simulated environment and users. It drives user activity and connects simulated behavior to the ML system.

**UserSimulator**

Models user reactions to recommendations, currently including:

- recommendation exposure
- clicks
- watches
- ratings

The behavior model is intentionally replaceable and will become more sophisticated as SimPipeline evolves.

### Online System

**Kafka**

Acts as the event transport layer for recommendation and behavioral events.

**Flink**

Processes the event stream and represents the streaming/online feature-processing layer.

**Redis**

Provides online storage used by the retrieval system.

**TwoTowerRetrieval**

Performs candidate retrieval using:

- a user tower
- an item tower
- an HNSW ANN index
- online feature/state access

**ModelServer**

Serves recommendations through the retrieval system.

### Offline System

**DataLake**

Stores historical events used for offline processing and retraining.

**Spark**

Represents the offline computation layer.

**Offline Feature Pipeline**

Builds offline features used by training.

**Offline Feature Store**

Stores offline feature data.

**Training Dataset Generator**

Produces training data from the available historical/features data.

**Two-Tower Trainer**

Trains the user and item towers.

**Embedding Generator**

Produces embeddings used by the retrieval system.

**HNSW Index**

Builds the approximate nearest-neighbor index over item embeddings.

## Retraining

The system periodically runs the offline pipeline over accumulated DataLake events.

The resulting artifacts include new model towers and item embeddings. A new HNSW index is constructed from the item embeddings, after which the online retriever is updated:

    new User Tower
           │
           ▼
    TwoTowerRetrieval

    new Item Tower
           │
           ▼
    TwoTowerRetrieval

    new Item Embeddings
           │
           ▼
       HNSW Index
           │
           ▼
    TwoTowerRetrieval

This models an important production property of ML systems: **the model serving the online system changes over time as new data is collected and new models are trained.**

## Why SimPipeline

Most ML education focuses on models and algorithms.

Production ML systems are larger than the model.

A recommendation system depends on data collection, event transport, stream processing, feature computation, storage, retrieval, serving, training, model updates, and the interaction between all of them.

SimPipeline is designed to let users build, run, break, diagnose, and improve simulated ML systems.

The eventual learning experience will turn the system into a game:

1. Build an ML system from components.
2. Run it.
3. Observe its behavior and metrics.
4. Encounter realistic production problem scenarios.
5. Diagnose the underlying system issue.
6. Change the architecture, configuration, data, features, model, or infrastructure.
7. Run the system again.
8. Observe the technical and business consequences.

The same component model can then be used to construct different ML systems rather than creating a separate simulator for every scenario.

## Roadmap

The current recommendation system is the foundation for a much broader simulation environment.

Planned development includes:

- more realistic feature computation and serving
- realistic data distributions and drift
- stale features
- latency and throughput constraints
- failures and degraded services
- backpressure and resource constraints
- model and data quality problems
- training/serving skew
- more realistic retrieval and ranking
- monitoring and observability
- experiment and deployment workflows
- richer production incidents
- interchangeable ML infrastructure components
- additional ML system types such as fraud detection
- reusable scenarios and system configurations
- a game layer built on top of the simulation

## Project Status

SimPipeline is under active development.

The current system already runs an end-to-end simulated recommendation workload with online interaction, event collection, retrieval, and periodic offline retraining. The current implementation is intentionally simpler than a real production platform; the next stage is increasing the fidelity of each component and the interactions between them.

## License

SimPipeline is licensed under the **PolyForm Noncommercial License 1.0.0**.

The source is publicly available for permitted noncommercial use, subject to the terms of the license. Commercial use requires separate permission.

Copyright © 2026 Primoz Ravbar.
