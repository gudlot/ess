from ess.v20.wfm.frames_analytical import frames_analytical
import scipp as sc
import numpy as np

def _to_angular_frequency(f):
    return (2.0 * np.pi * sc.units.rad) * f

def test_frames_analytical_one_chopper_one_cutout():
    instrument = sc.Dataset()
    # single pixel set 10m down the beam
    instrument['position'] = sc.vector(value=[0,0,10], unit=sc.units.m)
    # single chopper set 5m down the beam
    z_offset = sc.array(dims=['chopper'], values=[5.0], unit=sc.units.m)
    no_offset = sc.zeros(dims=['chopper'], shape=[1], unit=sc.units.m)
    instrument['distance'] = sc.geometry.position(no_offset, no_offset, z_offset)
    # Arbitrary pulse length of 10us in duration
    pulse_length = 10 * sc.units.us
    instrument['pulse_length'] = pulse_length

    # We now setout to engineer a single cutout to deliver a 5us sub-pulse.
    # We do this by setting a single cutout of 45 deg (pi/8) and then calculating rotation speed
    window_opening_t = sc.to_unit(5.0 * sc.units.us, sc.units.s)
    window_size = np.pi / 4 # choose a windows size 45 degrees
    ratio_of_window = window_size / (np.pi * 2)  # ( 1/8th of 2pi)
    chopper_frequency = ratio_of_window / window_opening_t  # Required operational frequency of chopper

    instrument['angular_frequency'] = _to_angular_frequency(
        sc.broadcast(chopper_frequency, dims=['chopper'], shape=[1]))

    instrument['phase'] = sc.array(dims=['chopper'],
                 values=[np.pi/8],
                 unit=sc.units.rad)


    # window opening angle
    instrument['frame_start'] = sc.array(dims=['frame'],
                                                     values=[0], unit=sc.units.rad) \
                                * sc.ones(dims=['chopper'], shape=[1])
    # window closing angle
    instrument['frame_end'] = sc.array(dims=['frame'],
                                         values=[window_size], unit=sc.units.rad) \
                                * sc.ones(dims=['chopper'], shape=[1])

    instrument['tdc'] = sc.zeros(dims=['chopper'], shape=[1])
    instrument['choppers'] = sc.array(dims=['chopper'],
                                      values=['chopper'])

    offset = 5.0 * sc.units.us # start of pulse t offset from 0
    frames = frames_analytical(instrument, offset=offset)
    # Following results best understood sketching simple time-distance diagram of above.
    assert sc.identical(frames['left_edges'].data, sc.array(dims=['frame'], values=[-5.0], unit=sc.units.us) + offset)
    assert sc.identical(frames['right_edges'].data, sc.array(dims=['frame'], values=[15.0], unit=sc.units.us) + offset)
    assert sc.identical(frames['shifts'].data, sc.array(dims=['frame'], values=[5.0], unit=sc.units.us) + offset)
