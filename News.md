## 2026-10-01

### Spark Simulation

Implemented the first complete Spark execution model in SimPipeline.

The Spark service now creates and coordinates a simulated **Application, Driver, Job Scheduler, Task Scheduler, Executor, Job, Stage, and Task**. Each component has its own lifecycle state, metrics, timestamps, and logs, with failures propagated through the execution hierarchy.

The execution flow is now:

**Application → Driver → Scheduler → Job → Stage → TaskScheduler → Executor → Task**

For the offline feature job, the Spark service creates a Job and Stage containing a Task. The Task holds the actual workload, while the Executor is responsible only for executing the Task. The Task currently invokes:

`feature_pipeline.process(events)`

The existing feature pipeline runs inside the simulated Spark execution path. The entire `events` dataset is currently processed as a single task.

The current implementation is a single-node, single-task model:

**1 Application → 1 Job → 1 Stage → 1 Task → 1 Executor**

This establishes the execution architecture before introducing distributed data processing.

**Next:** partition the input dataset and create one Task per partition, providing the foundation for parallel execution and the later implementation of Spark DAGs, stage boundaries, shuffle, scheduling, and task retries.
