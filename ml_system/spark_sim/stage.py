from enum import Enum

from ml_system.spark_sim.task import TaskState


class StageState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class SparkStage:

    def __init__(
        self,
        stage_id,
        name,
        tasks=None,
        dependencies=None
    ):
        self.stage_id = stage_id
        self.name = name

        self.state = StageState.PENDING

        self.tasks = tasks or []
        self.dependencies = dependencies or []

        self.current_task = None

        self.start_time = None
        self.end_time = None

        self.metrics = {
            "tasks_submitted": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
        }

        self.logs = []

    def add_task(self, task):

        self.tasks.append(task)

        self.metrics["tasks_submitted"] += 1

        self.log(
            f"Task {task.task_id} added to stage {self.stage_id}"
        )

    def start(self, clock):

        if self.state != StageState.PENDING:
            raise RuntimeError(
                f"Cannot start stage in state {self.state.value}"
            )

        self.state = StageState.RUNNING
        self.start_time = clock.now

        self.log(
            f"Stage {self.stage_id} started at tick {clock.now}"
        )

    def start_task(self, task):

        if self.state != StageState.RUNNING:
            raise RuntimeError(
                f"Cannot start task while stage is "
                f"in state {self.state.value}"
            )

        if task not in self.tasks:
            raise RuntimeError(
                f"Task {task.task_id} does not belong "
                f"to stage {self.stage_id}"
            )

        self.current_task = task

    def complete_task(self, task):

        if task not in self.tasks:
            raise RuntimeError(
                f"Task {task.task_id} does not belong "
                f"to stage {self.stage_id}"
            )

        if task.state != TaskState.SUCCEEDED:
            raise RuntimeError(
                f"Cannot complete task {task.task_id} "
                f"because task is in state {task.state.value}"
            )

        self.metrics["tasks_succeeded"] += 1

        self.log(
            f"Task {task.task_id} completed"
        )

        if self.current_task is task:
            self.current_task = None

    def fail_task(self, task):

        if task not in self.tasks:
            raise RuntimeError(
                f"Task {task.task_id} does not belong "
                f"to stage {self.stage_id}"
            )

        if task.state != TaskState.FAILED:
            raise RuntimeError(
                f"Cannot fail task {task.task_id} "
                f"because task is in state {task.state.value}"
            )

        self.metrics["tasks_failed"] += 1

        self.log(
            f"Task {task.task_id} failed"
        )

        if self.current_task is task:
            self.current_task = None

    def complete(self, clock):

        if self.state != StageState.RUNNING:
            raise RuntimeError(
                f"Cannot complete stage in state {self.state.value}"
            )

        if not self.tasks:
            raise RuntimeError(
                f"Cannot complete stage {self.stage_id} "
                "without tasks"
            )

        if any(
            task.state != TaskState.SUCCEEDED
            for task in self.tasks
        ):
            raise RuntimeError(
                f"Cannot complete stage {self.stage_id}: "
                "not all tasks succeeded"
            )

        self.state = StageState.SUCCEEDED
        self.end_time = clock.now

        self.log(
            f"Stage {self.stage_id} completed at tick {clock.now}"
        )

    def fail(self, clock, error):

        if self.state != StageState.RUNNING:
            raise RuntimeError(
                f"Cannot fail stage in state {self.state.value}"
            )

        self.state = StageState.FAILED
        self.end_time = clock.now

        self.log(
            f"Stage {self.stage_id} failed: {error}"
        )

    def log(self, message):

        self.logs.append(message)
