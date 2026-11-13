"""Redis: the online key-value store, with memory limits, eviction and TTLs.

The defaults behave exactly like the original dict: unlimited keys, no
expiry. The operational knobs, all off by default:

- max_keys (0 = unlimited) with an eviction policy, applied on every write
  that grows the store past the limit:
    "allkeys-lru"  evict the least recently used key (reads and writes count),
    "volatile-ttl" evict the key closest to expiring among keys with a TTL
                   (falls back to LRU when none has one),
    "noeviction"   refuse the write (counted in write_errors).
  Keys in PINNED (shared objects that components hold references to, such
  as Flink's movie signals and the preference model) are never evicted.
- ttl_ticks: feature name -> ticks a value lives after its last write
  (expired values read as missing: serving falls back to its defaults).
- Stats: hits, misses, evictions, expirations, write errors.

`store` stays a plain dict keyed by (feature, entity_id); the clock comes
from the simulation the service is started in.
"""

from collections import OrderedDict

from simulation.service import Service

EVICTION_POLICIES = ("allkeys-lru", "volatile-ttl", "noeviction")
PINNED = {"item_signals", "preference_model"}


class Redis(Service):

    def __init__(self, max_keys=0, eviction="allkeys-lru", ttl_ticks=None):
        super().__init__()
        self.store = {}

        # feature name -> fn(value) applied on read. Empty in normal
        # operation; fault injection uses it to corrupt what serving sees.
        self.read_transforms = {}

        if eviction not in EVICTION_POLICIES:
            raise ValueError("eviction must be one of %s" % (EVICTION_POLICIES,))
        self.max_keys = max_keys
        self.eviction = eviction
        self.ttl_ticks = dict(ttl_ticks or {})
        self.recency = OrderedDict()    # key -> None, least recently used first (only with max_keys)
        self.written = {}               # key -> tick of last write (only for features with a TTL)
        self.sim = None
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "expirations": 0, "write_errors": 0}

    def start(self, sim):
        super().start(sim)
        self.sim = sim

    def _now(self):
        return self.sim.clock.now if self.sim is not None else 0

    # -- writes ----------------------------------------------------------------

    def receive(self, feature):

        if "user_id" in feature:
            key = (
                feature["feature"],
                feature["user_id"]
            )

        elif "movie_id" in feature:
            key = (
                feature["feature"],
                feature["movie_id"]
            )

        self._write(key, feature["value"])

    def set(self, feature, entity_id, value):
        self._write((feature, entity_id), value)

    def _write(self, key, value):
        if self.max_keys and key not in self.store and len(self.store) >= self.max_keys:
            if not self._evict_one():
                self.stats["write_errors"] += 1
                return
        self.store[key] = value
        if self.ttl_ticks.get(key[0]):
            self.written[key] = self._now()
        if self.max_keys:
            self.recency[key] = None
            self.recency.move_to_end(key)

    def _evict_one(self):
        if self.eviction == "noeviction":
            return False
        victim = None
        if self.eviction == "volatile-ttl":
            now = self._now()
            best = None
            for key, tick in self.written.items():
                if key[0] in PINNED:
                    continue
                left = tick + self.ttl_ticks.get(key[0], 0) - now
                if best is None or left < best:
                    best, victim = left, key
        if victim is None:
            for key in self.recency:
                if key[0] not in PINNED:
                    victim = key
                    break
        if victim is None:
            return False
        self._delete(victim)
        self.stats["evictions"] += 1
        return True

    def _delete(self, key):
        self.store.pop(key, None)
        self.recency.pop(key, None)
        self.written.pop(key, None)

    def enforce(self):
        """Apply a lowered max_keys now (evicting, or nothing with noeviction)."""
        if not self.max_keys:
            return
        for key in self.store:
            if key not in self.recency:
                self.recency[key] = None
        while len(self.store) > self.max_keys and self._evict_one():
            pass

    # -- reads -----------------------------------------------------------------

    def get(self, feature, entity_id):
        key = (feature, entity_id)
        value = self.store.get(key)
        if value is not None and self.ttl_ticks.get(feature):
            if self._now() - self.written.get(key, self._now()) > self.ttl_ticks[feature]:
                self._delete(key)
                self.stats["expirations"] += 1
                value = None
        if value is None:
            self.stats["misses"] += 1
        else:
            self.stats["hits"] += 1
            if self.max_keys and key in self.recency:
                self.recency.move_to_end(key)
        transform = self.read_transforms.get(feature)
        if transform is not None and value is not None:
            value = transform(value)
        return value

    # -- reporting -------------------------------------------------------------

    def describe(self):
        reads = self.stats["hits"] + self.stats["misses"]
        return {"keys": len(self.store), "max_keys": self.max_keys, "eviction": self.eviction,
                "ttl_ticks": dict(self.ttl_ticks), **self.stats,
                "hit_rate": self.stats["hits"] / reads if reads else None}
