import scipp as sc
import numpy as np

def _to_angular_frequency(f):
    return (2.0 * np.pi) * f


def allclose(x, y):
    return sc.all(sc.isclose(x, y)).value


def _chopper_ang_freq(window_opening_t, window_size):
    ratio_of_window = window_size / (np.pi * 2)
    # Required operational frequency of chopper
    chopper_frequency = _to_angular_frequency(ratio_of_window /
                                              window_opening_t)
    return chopper_frequency


# Single chopper single cutout
def _single_chopper_beamline(window_opening_t,
                             pulse_length,
                             window_size=np.pi / 4 * sc.units.rad,
                             phase=None):
    instrument = sc.Dataset()
    # single pixel set 10m down the beam
    instrument['position'] = sc.vector(value=[0, 0, 10], unit=sc.units.m)
    # single chopper set 5m down the beam
    z_offset = sc.array(dims=['chopper'], values=[5.0], unit=sc.units.m)
    no_offset = sc.zeros(dims=['chopper'], shape=[1], unit=sc.units.m)
    instrument['distance'] = sc.geometry.position(no_offset, no_offset,
                                                  z_offset)
    # Arbitrary pulse length of 10us in duration
    instrument['pulse_length'] = pulse_length

    # We now set out to engineer a single cutout to deliver a specified
    # length sub-pulse.
    window_opening_t = sc.to_unit(window_opening_t, sc.units.s)
    chopper_ang_frequency = _chopper_ang_freq(window_opening_t, window_size)
    instrument['angular_frequency'] = sc.broadcast(chopper_ang_frequency,
                                                   dims=['chopper'],
                                                   shape=[1])

    if phase is None:
        blind_t = sc.to_unit(pulse_length, sc.units.s) - window_opening_t
        # For calc simplicity to center cutout opening over center of pulse
        phase_offset_t = blind_t / 2
        phase = phase_offset_t * chopper_ang_frequency
        instrument['phase'] = sc.array(dims=['chopper'],
                                       values=[phase.value],
                                       unit=sc.units.rad)
    else:
        instrument['phase'] = sc.broadcast(phase, dims=['chopper'], shape=[1])

    # window opening angle
    instrument['frame_start'] = sc.array(
        dims=['frame'], values=[0], unit=sc.units.rad) * sc.ones(
        dims=['chopper'], shape=[1])
    # window closing angle
    instrument['frame_end'] = sc.ones(dims=['frame'], shape=[1]) * sc.ones(
        dims=['chopper'], shape=[1]) * window_size

    instrument['choppers'] = sc.array(dims=['chopper'], values=['chopper'])
    return instrument
