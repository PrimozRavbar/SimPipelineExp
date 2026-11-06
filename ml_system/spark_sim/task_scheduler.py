from enum import Enum

from ml_system.spark_sim.task import TaskState


class TaskSchedulerState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class TaskScheduler:

    def __init__(self, scheduler_id):

        self.scheduler_id = scheduler_id

        self.state = TaskSchedulerState.IDLE

        self.pending_tasks = []
        self.running_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []

        self.executors = []

        self.metrics = {
            "tasks_submitted": 0,
            "tasks_started": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
        }

        self.logs = []

    def start(self):

        self.state = TaskSchedulerState.RUNNING

        self.log(
            f"Task scheduler {self.scheduler_id} started"
        )

    def add_executor(self, executor):

        self.executors.append(executor)

        self.log(
            f"Executor {executor.executor_id} registered"
        )

    def submit(self, task):

        if self.state != TaskSchedulerState.RUNNING:
            raise RuntimeError(
                "Cannot submit task to task scheduler "
                f"in state {self.state.value}"
            )

        self.pending_tasks.append(task)

        self.metrics["tasks_submitted"] += 1

        self.log(
            f"Task {task.task_id} submitted"
        )

    def schedule(self, clock):

        if self.state != TaskSchedulerState.RUNNING:
            return

        for executor in self.executors:

            if not self.pending_tasks:
                break

            task = self.pending_tasks[0]

            if not executor.can_run(task):
                continue

            self.pending_tasks.pop(0)
            self.running_tasks.append(task)

            self.metrics["tasks_started"] += 1

            executor.run_task(
                task,
                clock
            )

            self.running_tasks.remove(task)

            if task.state == TaskState.SUCCEEDED:

                self.task_completed(task)

            elif task.state == TaskState.FAILED:

                self.task_failed(task)

    def task_completed(self, task):

        self.completed_tasks.append(task)

        self.metrics["tasks_completed"] += 1

        self.log(
            f"Task {task.task_id} completed"
        )

    def task_failed(self, task):

        self.failed_tasks.append(task)

        self.metrics["tasks_failed"] += 1

        self.log(
            f"Task {task.task_id} failed"
        )

    def stop(self):

        self.state = TaskSchedulerState.STOPPED

        self.log(
            f"Task scheduler {self.scheduler_id} stopped"
        )

    def log(self, message):

        self.logs.append(message)
