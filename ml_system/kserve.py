"""KServe: the inference service in front of the ModelServer.

An InferenceService runs the model on replicas. Each replica handles
`replica_capacity` requests per tick (0 = unlimited, the default and the
original behaviour: every request is served by the model). With a capacity:

- requests beyond what the ready replicas can take in a tick are not
  served by the model: the ModelServer answers them from its fallback
  (popularity), and they are counted as overflow;
- the autoscaler (Knative-style, once per tick) sizes the deployment for
  last tick's traffic at `target_utilization`, between min and max
  replicas. New replicas take `cold_start_ticks` to become ready; extra
  replicas are removed after `scale_down_delay` ticks of lower demand;
- estimated latency grows with utilisation (an M/M/1-style 1 / (1 - u)
  factor on the base latency), for display only.

Uses no RNG.
"""

import math

from simulation.service import Service


class KServe(Service):

    def __init__(self, model_server, min_replicas=1, max_replicas=1, replica_capacity=0,
                 target_utilization=0.7, cold_start_ticks=3, scale_down_delay=20,
                 base_latency_ms=5.0):
        super().__init__()
        self.model_server = model_server
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.replica_capacity = replica_capacity
        self.target_utilization = target_utilization
        self.cold_start_ticks = cold_start_ticks
        self.scale_down_delay = scale_down_delay
        self.base_latency_ms = base_latency_ms
        self.ready = min_replicas
        self.starting = []          # ticks at which starting replicas become ready
        self.low_demand_ticks = 0
        self.requests = 0           # this tick
        self.overflow = 0           # this tick
        self.last = {"requests": 0, "overflow": 0, "utilization": 0.0}
        self.totals = {"requests": 0, "overflow": 0}
        self.replica_ticks = 0      # replica x ticks, for the bill

    def start(self, sim):
        super().start(sim)
        self.model_server.admission = self.admit

    # -- admission ----------------------------------------------------------------

    def capacity(self):
        return None if not self.replica_capacity else self.ready * self.replica_capacity

    def admit(self):
        """One request: True if a ready replica can take it this tick."""
        self.requests += 1
        cap = self.capacity()
        if cap is None or self.requests <= cap:
            return True
        self.overflow += 1
        return False

    # -- autoscaling (end of tick) -------------------------------------------------

    def process(self, sim):
        now = sim.clock.now
        cap = self.capacity()
        utilization = 0.0 if not cap else min(self.requests / cap, 1.0) if cap else 0.0
        self.last = {"requests": self.requests, "overflow": self.overflow, "utilization": utilization}
        self.totals["requests"] += self.requests
        self.totals["overflow"] += self.overflow
        self.replica_ticks += self.ready + len(self.starting)

        # Replicas that finished starting.
        ready_now = [t for t in self.starting if t <= now]
        self.starting = [t for t in self.starting if t > now]
        self.ready += len(ready_now)

        if self.replica_capacity:
            desired = math.ceil(self.requests / (self.replica_capacity * self.target_utilization)) or self.min_replicas
        else:
            desired = self.min_replicas
        desired = max(self.min_replicas, min(self.max_replicas, desired))
        total = self.ready + len(self.starting)
        if desired > total:
            self.starting += [now + self.cold_start_ticks] * (desired - total)
            self.low_demand_ticks = 0
        elif desired < self.ready:
            self.low_demand_ticks += 1
            if self.low_demand_ticks >= self.scale_down_delay:
                self.ready -= 1
                self.low_demand_ticks = 0
        else:
            self.low_demand_ticks = 0
        # Settings may have moved the bounds.
        self.ready = max(min(self.ready, self.max_replicas), 0)

        self.requests = 0
        self.overflow = 0

    # -- reporting -----------------------------------------------------------------

    def latency_ms(self):
        u = min(self.last["utilization"], 0.95)
        return self.base_latency_ms / (1.0 - u) if self.replica_capacity else self.base_latency_ms

    def describe(self):
        return {"ready_replicas": self.ready, "starting_replicas": len(self.starting),
                "min_replicas": self.min_replicas, "max_replicas": self.max_replicas,
                "replica_capacity": self.replica_capacity, "capacity": self.capacity(),
                "target_utilization": self.target_utilization, "cold_start_ticks": self.cold_start_ticks,
                "last_tick": dict(self.last), "totals": dict(self.totals),
                "overflow_share": self.totals["overflow"] / self.totals["requests"] if self.totals["requests"] else 0.0,
                "estimated_latency_ms": round(self.latency_ms(), 1)}
