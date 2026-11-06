class Clock:

    def __init__(self):
        self.tick_number = 0

    def tick(self):
        self.tick_number += 1

    @property
    def now(self):
        return self.tick_number