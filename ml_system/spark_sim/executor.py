from enum import Enum

from ml_system.spark_sim.task import TaskState


class ExecutorState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class Executor:

    def __init__(
        self,
        executor_id,
        cores=1,
        memory=1
    ):
        self.executor_id = executor_id

        self.cores = cores
        self.memory = memory

        self.state = ExecutorState.IDLE

        self.current_task = None

        self.metrics = {
            "tasks_started": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
        }

        self.logs = []

    def start(self):

        self.state = ExecutorState.IDLE

        self.log(
            f"Executor {self.executor_id} started"
        )

    def can_run(self, task):

        return (
            self.state == ExecutorState.IDLE
            and self.current_task is None
        )

    def run_task(self, task, clock):

        if not self.can_run(task):
            raise RuntimeError(
                f"Executor {self.executor_id} "
                "cannot run task"
            )

        self.current_task = task

        task.assign_executor(self)
        task.start(clock)

        self.state = ExecutorState.RUNNING

        self.metrics["tasks_started"] += 1

        self.log(
            f"Started task {task.task_id}"
        )

        task.execute(clock)

        if task.state == TaskState.SUCCEEDED:

            self.metrics["tasks_succeeded"] += 1

            self.log(
                f"Task {task.task_id} succeeded"
            )

        elif task.state == TaskState.FAILED:

            self.metrics["tasks_failed"] += 1

            self.log(
                f"Task {task.task_id} failed"
            )

        self.current_task = None
        self.state = ExecutorState.IDLE

    def fail(self):

        self.state = ExecutorState.FAILED

        self.log(
            f"Executor {self.executor_id} failed"
        )

    def stop(self):

        self.state = ExecutorState.STOPPED

        self.log(
            f"Executor {self.executor_id} stopped"
        )

    def log(self, message):

        self.logs.append(message)
