from enum import Enum


class DriverState(Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Driver:

    def __init__(self, driver_id, scheduler):

        self.driver_id = driver_id
        self.scheduler = scheduler

        self.state = DriverState.IDLE

        self.jobs = []

        self.current_job = None

        self.metrics = {
            "jobs_submitted": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
        }

        self.logs = []

    def start(self, clock):

        self.state = DriverState.RUNNING

        self.log(
            f"Driver {self.driver_id} started at "
            f"tick {clock.now}"
        )

    def submit_job(self, job):

        if self.state != DriverState.RUNNING:
            raise RuntimeError(
                "Cannot submit job to driver in "
                f"state {self.state.value}"
            )

        self.jobs.append(job)
        self.current_job = job

        self.metrics["jobs_submitted"] += 1

        self.log(
            f"Driver received job {job.job_id}"
        )

        self.scheduler.submit(job)

    def complete_job(self, job):

        self.metrics["jobs_completed"] += 1

        self.log(
            f"Driver completed job {job.job_id}"
        )

        if self.current_job is job:
            self.current_job = None

    def fail_job(self, job):

        self.metrics["jobs_failed"] += 1

        self.log(
            f"Driver failed job {job.job_id}"
        )

        if self.current_job is job:
            self.current_job = None

    def complete(self):

        self.state = DriverState.COMPLETED

        self.log(
            f"Driver {self.driver_id} completed"
        )

    def fail(self):

        self.state = DriverState.FAILED

        self.log(
            f"Driver {self.driver_id} failed"
        )

    def log(self, message):

        self.logs.append(message)
