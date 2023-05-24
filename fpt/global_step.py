# global_step.py
class GlobalStep:
    def __init__(self):
        self._step = 0

    def increment(self):
        self._step += 1

    def get(self):
        return self._step