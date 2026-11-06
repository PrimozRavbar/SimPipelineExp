from enum import Enum


class SchedulerState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class Scheduler:

    def __init__(self, scheduler_id):

        self.scheduler_id = scheduler_id

        self.state = SchedulerState.IDLE

        self.pending_jobs = []
        self.running_jobs = []
        self.completed_jobs = []
        self.failed_jobs = []

        self.metrics = {
            "jobs_submitted": 0,
            "jobs_started": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
        }

        self.logs = []

    def start(self):

        self.state = SchedulerState.RUNNING

        self.log(
            f"Scheduler {self.scheduler_id} started"
        )

    def submit(self, job):

        if self.state != SchedulerState.RUNNING:
            raise RuntimeError(
                "Cannot submit job to scheduler in "
                f"state {self.state.value}"
            )

        self.pending_jobs.append(job)

        self.metrics["jobs_submitted"] += 1

        self.log(
            f"Job {job.job_id} added to scheduler queue"
        )

    def schedule_next(self):

        if self.state != SchedulerState.RUNNING:
            return None

        if not self.pending_jobs:
            return None

        job = self.pending_jobs.pop(0)

        self.running_jobs.append(job)

        self.metrics["jobs_started"] += 1

        self.log(
            f"Job {job.job_id} scheduled"
        )

        return job

    def complete(self, job):

        if job in self.running_jobs:
            self.running_jobs.remove(job)

        self.completed_jobs.append(job)

        self.metrics["jobs_completed"] += 1

        self.log(
            f"Job {job.job_id} completed"
        )

    def fail(self, job):

        if job in self.running_jobs:
            self.running_jobs.remove(job)

        self.failed_jobs.append(job)

        self.metrics["jobs_failed"] += 1

        self.log(
            f"Job {job.job_id} failed"
        )

    def stop(self):

        self.state = SchedulerState.STOPPED

        self.log(
            f"Scheduler {self.scheduler_id} stopped"
        )

    def log(self, message):

        self.logs.append(message)
