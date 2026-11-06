from simulation.service import Service

from ml_system.spark_sim.application import SparkApplication
from ml_system.spark_sim.driver import Driver
from ml_system.spark_sim.scheduler import Scheduler
from ml_system.spark_sim.job import SparkJob
from ml_system.spark_sim.stage import SparkStage
from ml_system.spark_sim.task import SparkTask
from ml_system.spark_sim.task_scheduler import TaskScheduler
from ml_system.spark_sim.executor import Executor


class Spark(Service):

    def __init__(self):
        super().__init__()

        self.sim = None
        self.application = None

        self.scheduler = None
        self.task_scheduler = None
        self.executor = None

        self.next_application_id = 1
        self.next_driver_id = 1
        self.next_scheduler_id = 1
        self.next_task_scheduler_id = 1
        self.next_executor_id = 1
        self.next_job_id = 1
        self.next_stage_id = 1
        self.next_task_id = 1

    def start(self, sim):

        super().start(sim)

        self.sim = sim

    def run_offline_feature_job(
        self,
        events,
        feature_pipeline
    ):

        self.scheduler = Scheduler(
            scheduler_id=self.next_scheduler_id
        )
        self.next_scheduler_id += 1
        self.scheduler.start()

        self.task_scheduler = TaskScheduler(
            scheduler_id=self.next_task_scheduler_id
        )
        self.next_task_scheduler_id += 1
        self.task_scheduler.start()

        self.executor = Executor(
            executor_id=self.next_executor_id
        )
        self.next_executor_id += 1
        self.executor.start()

        self.task_scheduler.add_executor(
            self.executor
        )

        driver = Driver(
            driver_id=self.next_driver_id,
            scheduler=self.scheduler
        )
        self.next_driver_id += 1

        self.application = SparkApplication(
            application_id=self.next_application_id,
            name="offline_feature_job",
            driver=driver
        )
        self.next_application_id += 1

        self.application.start(self.sim.clock)

        job = SparkJob(
            job_id=self.next_job_id,
            name="offline_feature_job"
        )
        self.next_job_id += 1

        self.application.submit_job(job)

        stage = SparkStage(
            stage_id=self.next_stage_id,
            name="offline_feature_stage"
        )
        self.next_stage_id += 1

        job.add_stage(stage)

        task = SparkTask(
            task_id=self.next_task_id,
            stage_id=stage.stage_id,
            work=lambda: feature_pipeline.process(events)
        )
        self.next_task_id += 1

        stage.add_task(task)

        scheduled_job = self.scheduler.schedule_next()

        if scheduled_job is None:
            raise RuntimeError(
                "Spark scheduler failed to schedule job"
            )

        job.start(self.sim.clock)
        job.start_stage(stage)

        stage.start(self.sim.clock)
        stage.start_task(task)

        self.task_scheduler.submit(task)
        self.task_scheduler.schedule(self.sim.clock)

        if task in self.task_scheduler.completed_tasks:

            stage.complete_task(task)
            stage.complete(self.sim.clock)

            job.complete_stage(stage)
            job.complete(self.sim.clock)

            self.scheduler.complete(job)
            driver.complete_job(job)

            self.application.complete_job(job)
            self.application.complete(self.sim.clock)

        elif task in self.task_scheduler.failed_tasks:

            error = task.error

            stage.fail_task(task)
            stage.fail(self.sim.clock, error)

            job.fail_stage(stage)
            job.fail(self.sim.clock, error)

            self.scheduler.fail(job)
            driver.fail_job(job)

            self.application.fail_job(job, error)
            self.application.fail(self.sim.clock, error)

            raise error

        else:

            raise RuntimeError(
                f"Task {task.task_id} has no terminal outcome"
            )
