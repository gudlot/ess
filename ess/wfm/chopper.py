import numpy as np


def deg_to_rad(x):
    return x * np.pi / 180.0


class Chopper:
    def __init__(self,
                 frequency=0,
                 openings=None,
                 distance=0,
                 phase=0,
                 unit="rad",
                 name=""):
        # openings is list. First entry is start angle of the first cut-out
        # second entry is end angle of first cut-out, etc.
        self.frequency = frequency
        self.openings = openings
        self.omega = 2.0 * np.pi * frequency
        self.distance = distance
        self.phase = phase
        if unit == "deg":
            self.openings = deg_to_rad(self.openings)
            self.phase = deg_to_rad(self.phase)
        self.name = name
