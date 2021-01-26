import numpy as np
from os.path import join
from .chopper import Chopper


def setup(file_handle=None, detector_positions=None):
    """
    :param input_file: An open hdf file handle
    """
    time_units = {"us": 1.0e6, "ns": 1.0e9}

    choppers = dict()
    info = dict()

    chopper_entries = {"WFM1": "/entry/WFM1", "WFM2": "/entry/WFM2"}

    frequency = 70.0
    openings = np.array([
        83.71, 94.7, 140.49, 155.79, 193.26, 212.56, 242.32, 265.33, 287.91,
        314.37, 330.3, 360.0
    ]) + 15.0
    if file_handle is not None:
        tdc = join(chopper_entries["WFM1"], "top_dead_center/time")
        if tdc in file_handle:
            timings = np.array(file_handle[tdc][...],
                               dtype=np.float64,
                               copy=True)
            frequency = np.mean(
                time_units[file_handle[tdc].attrs["units"].decode("utf-8")] /
                np.ediff1d(timings))
        slits = join(chopper_entries["WFM1"], "slit_edges")
        if slits in file_handle:
            openings = np.array(file_handle[slits][...],
                                dtype=np.float64,
                                copy=True)

    choppers["WFM1"] = Chopper(frequency=frequency,
                               openings=openings,
                               phase=47.10,
                               distance=6.6,
                               unit="deg",
                               name="WFM1")

    frequency = 70.0
    openings = np.array([
        65.04, 76.03, 126.1, 141.4, 182.88, 202.18, 235.67, 254.97, 284.73,
        307.74, 330.00, 360.0
    ]) + 15.0
    if file_handle is not None:
        tdc = join(chopper_entries["WFM2"], "top_dead_center/time")
        if tdc in file_handle:
            timings = np.array(file_handle[tdc][...],
                               dtype=np.float64,
                               copy=True)
            frequency = np.mean(
                time_units[file_handle[tdc].attrs["units"].decode("utf-8")] /
                np.ediff1d(timings))
        slits = join(chopper_entries["WFM2"], "slit_edges")
        if slits in file_handle:
            openings = np.array(file_handle[slits][...],
                                dtype=np.float64,
                                copy=True)
    choppers["WFM2"] = Chopper(frequency=frequency,
                               openings=openings,
                               phase=76.76,
                               distance=7.1,
                               unit="deg",
                               name="WFM2")

    choppers["FOL1"] = Chopper(frequency=56,
                               openings=np.array([
                                   64.35, 84.99, 125.05, 148.29, 183.41, 205.22,
                                   236.4, 254.27, 287.04, 302.8, 335.53, 360.0
                               ]) + 15.0,
                               phase=62.40,
                               distance=8.8,
                               unit="deg",
                               name="Frame-overlap 1")

    choppers["FOL2"] = Chopper(frequency=28,
                               openings=np.array([
                                   79.78, 116.38, 136.41, 172.47, 191.73, 221.94,
                                   240.81, 267.69, 287.13, 311.69, 330.89, 360.0
                               ]) + 15.0,
                               phase=12.27,
                               distance=15.9,
                               unit="deg",
                               name="Frame-overlap 2")

    # choppers["FOL1"] = Chopper(frequency=56,
    #                            openings=np.array([
    #                                74.6, 95.2, 139.6, 162.8, 194.3, 216.1,
    #                                245.3, 263.1, 294.8, 310.5, 347.2, 371.6
    #                            ]),
    #                            phase=62.40,
    #                            distance=8.8,
    #                            unit="deg",
    #                            name="Frame-overlap 1")

    # choppers["FOL2"] = Chopper(frequency=28,
    #                            openings=np.array([
    #                                98., 134.6, 154., 190.06, 206.8, 237.01,
    #                                254., 280.88, 299., 323.56, 344.65, 373.76
    #                            ]),
    #                            phase=12.27,
    #                            distance=15.9,
    #                            unit="deg",
    #                            name="Frame-overlap 2")

    # Number of frames
    info["nframes"] = 6

    # Length of pulse
    info["pulse_length"] = 2.86e-03

    # Position of detectors
    info["detector_positions"] = detector_positions
    # info["detector_positions"] = {"DENEX": 30.6, "monitor": 25.7}
    # info["detector_positions"] = {"DENEX": 28.42, "monitor": 25.6}

    # Midpoint between WFM choppers which acts as new source distance for stitched data
    info["wfm_choppers_midpoint"] = 0.5 * (choppers["WFM1"].distance +
                                           choppers["WFM2"].distance)

    return {"info": info, "choppers": choppers}
