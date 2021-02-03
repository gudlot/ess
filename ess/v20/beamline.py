import numpy as np
import scipp as sc


def _deg_to_rad(x):
    """
    Convert degrees to radians.
    """
    return x * (np.pi * sc.units.rad / 180.0)


def _to_angular_frequency(f):
    """
    Convert frequency in Hz to angular frequency.
    """
    return (2.0 * np.pi * sc.units.rad) * (f / (1.0 * sc.units.s))


def beamline():
    """
    Create V20 chopper cascade and component positions.
    Chopper opening angles taken from Woracek et al. (2016)
    https://doi.org/10.1016/j.nima.2016.09.034
    """

    ds = sc.Dataset()

    ds["choppers"] = sc.array(
        dims=["chopper"],
        values=["WFM1", "WFM2", "frame-overlap-1", "frame-overlap-2"])

    ds["angular_frequency"] = _to_angular_frequency(
        sc.array(dims=["chopper"], values=[70.0, 70.0, 56.0, 28.0]))

    ds["phase"] = _deg_to_rad(
        sc.array(dims=["chopper"], values=[47.10, 76.76, 62.40, 12.27]))

    ds["distance"] = sc.array(dims=["chopper"],
                              values=[[0, 0, 6.6], [0, 0, 7.1], [0, 0, 8.8],
                                      [0, 0, 15.9]],
                              unit=sc.units.m,
                              dtype=sc.dtype.vector_3_float64)

    ds["frame_start"] = _deg_to_rad(
        sc.array(
            dims=["chopper", "frame"],
            values=np.concatenate([
                np.array([83.71, 140.49, 193.26, 242.32, 287.91, 330.3]) +
                15.0,
                np.array([65.04, 126.1, 182.88, 235.67, 284.73, 330.00]) +
                15.0,
                np.array([64.35, 125.05, 183.41, 236.4, 287.04, 335.53]) +
                15.0,
                np.array([79.78, 136.41, 191.73, 240.81, 287.13, 330.89]) +
                15.0
            ]).reshape(4, 6)))

    ds["frame_end"] = _deg_to_rad(
        sc.array(
            dims=["chopper", "frame"],
            values=np.concatenate([
                np.array([94.7, 155.79, 212.56, 265.33, 314.37, 360.0]) + 15.0,
                np.array([76.03, 141.4, 202.18, 254.97, 307.74, 360.0]) + 15.0,
                np.array([84.99, 148.29, 205.22, 254.27, 302.8, 360.0]) + 15.0,
                np.array([116.38, 172.47, 221.94, 267.69, 311.69, 360.0]) +
                15.0
            ]).reshape(4, 6)))

    # Length of pulse
    ds["pulse_length"] = 2.86e+03 * sc.units.us

    return ds
