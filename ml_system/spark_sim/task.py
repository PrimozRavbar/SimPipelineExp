from enum import Enum


class TaskState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class SparkTask:

    def __init__(
        self,
        task_id,
        stage_id,
        work
    ):
        self.task_id = task_id
        self.stage_id = stage_id
        self.work = work

        self.state = TaskState.PENDING

        self.executor = None

        self.attempt = 0
        self.max_attempts = 1

        self.start_time = None
        self.end_time = None

        self.result = None
        self.error = None

        self.metrics = {
            "attempts": 0,
        }

        self.logs = []

    def assign_executor(self, executor):

        if self.executor is not None:
            raise RuntimeError(
                f"Task {self.task_id} already has an executor"
            )

        self.executor = executor

        self.log(
            f"Task {self.task_id} assigned to "
            f"executor {executor.executor_id}"
        )

    def start(self, clock):

        if self.state != TaskState.PENDING:
            raise RuntimeError(
                f"Cannot start task in state {self.state.value}"
            )

        if self.executor is None:
            raise RuntimeError(
                f"Task {self.task_id} has no executor"
            )

        self.state = TaskState.RUNNING
        self.attempt += 1
        self.metrics["attempts"] += 1
        self.start_time = clock.now

        self.log(
            f"Task {self.task_id} attempt "
            f"{self.attempt} started at tick {clock.now}"
        )

    def execute(self, clock):

        if self.state != TaskState.RUNNING:
            raise RuntimeError(
                f"Cannot execute task in state {self.state.value}"
            )

        try:
            self.result = self.work()
            self.complete(clock)

        except Exception as error:
            self.fail(clock, error)

    def complete(self, clock):

        if self.state != TaskState.RUNNING:
            raise RuntimeError(
                f"Cannot complete task in state {self.state.value}"
            )

        self.state = TaskState.SUCCEEDED
        self.end_time = clock.now

        self.log(
            f"Task {self.task_id} succeeded"
        )

    def fail(self, clock, error=None):

        if self.state != TaskState.RUNNING:
            raise RuntimeError(
                f"Cannot fail task in state {self.state.value}"
            )

        self.state = TaskState.FAILED
        self.end_time = clock.now

        if error is not None:
            self.error = error

        self.log(
            f"Task {self.task_id} failed: {self.error}"
        )

    def log(self, message):

        self.logs.append(message)
