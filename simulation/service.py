from enum import Enum


class ServiceStatus(Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class Service:

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__
        self.status = ServiceStatus.STOPPED

        self.inbox = []
        self.outbox = []

        self.metrics = {}
        self.logs = []
        self.config = {}

    def start(self, sim):
        self.status = ServiceStatus.RUNNING

    def stop(self, sim):
        self.status = ServiceStatus.STOPPED

    def receive(self, event):
        self.inbox.append(event)

    def process(self, sim):
        """Override in subclasses."""
        pass

    def emit(self):
        events = self.outbox
        self.outbox = []
        return events

    def tick(self, sim):
        if self.status != ServiceStatus.RUNNING:
            return

        self.process(sim)

        for event in self.emit():
            sim.publish(event)

    def log(self, message):
        self.logs.append(message)