"""The money budget (a simple first version).

Separate from the compute credits that pay for retrains and Spark runs:
cash pays for running the infrastructure, and clicks earn it.

- Running costs, charged every tick from the current configuration of each
  component (PRICES): Kafka brokers and partitions, Flink task slots,
  Redis memory (per 1000 keys), KServe replicas, the ANN index type and
  Feast (plus each materialization run).
- One-off costs (ONE_OFF): swapping a component for another (migration),
  rescaling Flink (savepoint + restart), repartitioning Kafka.
- Revenue: REVENUE_PER_CLICK for every click.

With the default setup the bill is about $5 a tick, so a model at ~30% CTR
with 20 users a tick roughly breaks even and a better one makes money.
Cash below zero opens an incident; nothing stops (yet).
"""

from ml_system.ann_indexes import INDEX_TYPES

START_CASH = 10000.0
REVENUE_PER_CLICK = 1.0

PRICES = {
    "kafka_broker": 0.5,
    "kafka_partition": 0.05,
    "flink_slot": 0.8,
    "redis_base": 0.2,
    "redis_per_1000_keys": 0.1,
    "kserve_replica": 1.0,
    "feast_base": 0.3,
    "feast_materialization": 2.0,
}

ONE_OFF = {
    "swap_ann_index": 500.0,
    "flink_rescale": 20.0,
    "kafka_repartition": 50.0,
}


class Economy:

    def __init__(self, start_cash=START_CASH):
        self.start_cash = start_cash
        self.cash = start_cash
        self.revenue = 0.0
        self.running = 0.0
        self.one_off = 0.0
        self.last_tick = {"revenue": 0.0, "cost": 0.0, "breakdown": {}}
        self.ledger = []               # (tick, amount, reason) for one-off charges

    def running_costs(self, system, settings, index_kind):
        kafka = system["sim"].get_service("Kafka")
        flink = system["sim"].get_service("Flink")
        redis = system["redis"]
        kserve = system["kserve"]
        return {
            "Kafka": kafka.brokers * PRICES["kafka_broker"] + kafka.partitions * PRICES["kafka_partition"],
            "Flink": flink.parallelism * PRICES["flink_slot"],
            "Redis": PRICES["redis_base"] + len(redis.store) / 1000.0 * PRICES["redis_per_1000_keys"],
            "KServe": (kserve.ready + len(kserve.starting)) * PRICES["kserve_replica"],
            "ANN index": INDEX_TYPES.get(index_kind, INDEX_TYPES["hnsw"])["price_per_tick"],
            "Feast": PRICES["feast_base"],
        }

    def tick(self, now, clicks, breakdown):
        revenue = clicks * REVENUE_PER_CLICK
        cost = sum(breakdown.values())
        self.revenue += revenue
        self.running += cost
        self.cash += revenue - cost
        self.last_tick = {"tick": now, "revenue": revenue, "cost": round(cost, 4),
                          "breakdown": {k: round(v, 4) for k, v in breakdown.items()}}

    def charge(self, now, amount, reason):
        self.cash -= amount
        self.one_off += amount
        self.ledger.append((now, amount, reason))

    def view(self):
        return {"cash": round(self.cash, 2), "start_cash": self.start_cash,
                "revenue": round(self.revenue, 2), "running_costs": round(self.running, 2),
                "one_off_costs": round(self.one_off, 2), "last_tick": self.last_tick,
                "revenue_per_click": REVENUE_PER_CLICK, "prices": PRICES, "one_off": ONE_OFF,
                "index_prices": {k: v["price_per_tick"] for k, v in INDEX_TYPES.items()},
                "ledger": [{"tick": t, "amount": a, "reason": r} for t, a, r in self.ledger[-20:]]}
