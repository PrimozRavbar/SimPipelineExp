from enum import Enum

from ml_system.spark_sim.stage import StageState


class JobState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class SparkJob:

    def __init__(
        self,
        job_id,
        name,
        stages=None
    ):
        self.job_id = job_id
        self.name = name

        self.state = JobState.PENDING

        self.stages = stages or []

        self.current_stage = None

        self.start_time = None
        self.end_time = None

        self.metrics = {
            "stages_submitted": 0,
            "stages_succeeded": 0,
            "stages_failed": 0,
        }

        self.logs = []

    def add_stage(self, stage):

        self.stages.append(stage)

        self.metrics["stages_submitted"] += 1

        self.log(
            f"Stage {stage.stage_id} added to job {self.job_id}"
        )

    def start_stage(self, stage):

        if self.state != JobState.RUNNING:
            raise RuntimeError(
                f"Cannot start stage while job is "
                f"in state {self.state.value}"
            )

        if stage not in self.stages:
            raise RuntimeError(
                f"Stage {stage.stage_id} does not belong "
                f"to job {self.job_id}"
            )

        self.current_stage = stage

    def complete_stage(self, stage):

        if stage not in self.stages:
            raise RuntimeError(
                f"Stage {stage.stage_id} does not belong "
                f"to job {self.job_id}"
            )

        if stage.state != StageState.SUCCEEDED:
            raise RuntimeError(
                f"Cannot complete stage {stage.stage_id} "
                f"because stage is in state {stage.state.value}"
            )

        self.metrics["stages_succeeded"] += 1

        self.log(
            f"Stage {stage.stage_id} completed"
        )

        if self.current_stage is stage:
            self.current_stage = None

    def fail_stage(self, stage):

        if stage not in self.stages:
            raise RuntimeError(
                f"Stage {stage.stage_id} does not belong "
                f"to job {self.job_id}"
            )

        if stage.state != StageState.FAILED:
            raise RuntimeError(
                f"Cannot fail stage {stage.stage_id} "
                f"because stage is in state {stage.state.value}"
            )

        self.metrics["stages_failed"] += 1

        self.log(
            f"Stage {stage.stage_id} failed"
        )

        if self.current_stage is stage:
            self.current_stage = None

    def start(self, clock):

        if self.state != JobState.PENDING:
            raise RuntimeError(
                f"Cannot start job in state {self.state.value}"
            )

        self.state = JobState.RUNNING
        self.start_time = clock.now

        self.log(
            f"Job {self.job_id} started at tick {clock.now}"
        )

    def complete(self, clock):

        if self.state != JobState.RUNNING:
            raise RuntimeError(
                f"Cannot complete job in state {self.state.value}"
            )

        if not self.stages:
            raise RuntimeError(
                f"Cannot complete job {self.job_id} "
                "without stages"
            )

        if any(
            stage.state != StageState.SUCCEEDED
            for stage in self.stages
        ):
            raise RuntimeError(
                f"Cannot complete job {self.job_id}: "
                "not all stages succeeded"
            )

        self.state = JobState.SUCCEEDED
        self.end_time = clock.now

        self.log(
            f"Job {self.job_id} completed at tick {clock.now}"
        )

    def fail(self, clock, error):

        if self.state != JobState.RUNNING:
            raise RuntimeError(
                f"Cannot fail job in state {self.state.value}"
            )

        self.state = JobState.FAILED
        self.end_time = clock.now

        self.log(
            f"Job {self.job_id} failed: {error}"
        )

    def log(self, message):

        self.logs.append(message)
