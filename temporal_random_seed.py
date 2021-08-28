import random

class TemporalRandomSeed:
    def __init__(self, seed):
        self.nop = seed is None
        if not self.nop:
            self.prev_state = random.getstate()
            random.seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        if not self.nop:
            random.setstate(self.prev_state)


if __name__ == "__main__":
    random.seed(1000)
    print(random.random())
    print(random.random())
    print(random.random())
    random.seed(1000)
    print(random.random())
    print(random.random())
    with TemporalRandomSeed(None):
        print(random.random())
        print(random.random())
    print(random.random())
