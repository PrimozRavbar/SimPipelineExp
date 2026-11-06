from enum import Enum


class ApplicationState(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SparkApplication:

    def __init__(
        self,
        application_id,
        name,
        driver
    ):
        self.application_id = application_id
        self.name = name
        self.driver = driver

        self.state = ApplicationState.CREATED

        self.jobs = []

        self.current_job = None

        self.start_time = None
        self.end_time = None

        self.metrics = {
            "jobs_submitted": 0,
            "jobs_succeeded": 0,
            "jobs_failed": 0
        }

        self.logs = []

    def start(self, clock):

        if self.state != ApplicationState.CREATED:
            raise RuntimeError(
                f"Cannot start application in state "
                f"{self.state.value}"
            )

        self.start_time = clock.now
        self.state = ApplicationState.RUNNING

        self.log(
            f"Application {self.application_id} started"
        )

        self.driver.start(clock)

    def submit_job(self, job):

        if self.state != ApplicationState.RUNNING:
            raise RuntimeError(
                "Cannot submit a job to an application "
                f"in state {self.state.value}"
            )

        self.jobs.append(job)
        self.current_job = job

        self.metrics["jobs_submitted"] += 1

        self.log(
            f"Job {job.job_id} submitted"
        )

        self.driver.submit_job(job)

    def complete_job(self, job):

        if job not in self.jobs:
            raise RuntimeError(
                f"Job {job.job_id} does not belong "
                f"to application {self.application_id}"
            )

        self.metrics["jobs_succeeded"] += 1

        self.log(
            f"Job {job.job_id} completed"
        )

        if self.current_job is job:
            self.current_job = None

    def fail_job(self, job, error):

        if job not in self.jobs:
            raise RuntimeError(
                f"Job {job.job_id} does not belong "
                f"to application {self.application_id}"
            )

        self.metrics["jobs_failed"] += 1

        self.log(
            f"Job {job.job_id} failed: {error}"
        )

        if self.current_job is job:
            self.current_job = None

    def complete(self, clock):

        if self.state != ApplicationState.RUNNING:
            raise RuntimeError(
                f"Cannot complete application in state "
                f"{self.state.value}"
            )

        self.state = ApplicationState.SUCCEEDED
        self.end_time = clock.now

        self.driver.complete()

        self.log(
            f"Application {self.application_id} completed"
        )

    def fail(self, clock, error):

        if self.state != ApplicationState.RUNNING:
            raise RuntimeError(
                f"Cannot fail application in state "
                f"{self.state.value}"
            )

        self.state = ApplicationState.FAILED
        self.end_time = clock.now

        self.driver.fail()

        self.log(
            f"Application {self.application_id} failed: {error}"
        )

    def cancel(self, clock):

        if self.state != ApplicationState.RUNNING:
            raise RuntimeError(
                f"Cannot cancel application in state "
                f"{self.state.value}"
            )

        self.state = ApplicationState.CANCELLED
        self.end_time = clock.now

        self.log(
            f"Application {self.application_id} cancelled"
        )

    def log(self, message):

        self.logs.append(message)
