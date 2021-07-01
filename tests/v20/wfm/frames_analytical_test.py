from ess.v20.wfm.frames_analytical import frames_analytical
import scipp as sc
import numpy as np


def _to_angular_frequency(f):
    return (2.0 * np.pi * sc.units.rad) * f


def allclose(x, y):
    return sc.all(sc.isclose(x, y)).value


def _single_chopper_beamline(window_opening_t, pulse_length, tdc=None):
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
    # length sub-pulse. We do this by setting a single cutout of 45 deg
    # (pi/8) and then calculating rotation speed
    window_opening_t = sc.to_unit(window_opening_t, sc.units.s)
    window_size = np.pi / 4  # choose a windows size 45 degrees
    ratio_of_window = window_size / (np.pi * 2)
    # Required operational frequency of chopper
    chopper_frequency = ratio_of_window / window_opening_t

    instrument['angular_frequency'] = _to_angular_frequency(
        sc.broadcast(chopper_frequency, dims=['chopper'], shape=[1]))

    blind_t = sc.to_unit(pulse_length, sc.units.s) - window_opening_t
    # For calculation simplicity to center cutout opening over center of pulse
    phase_offset_t = blind_t / 2
    phase = 2 * np.pi * phase_offset_t * chopper_frequency
    instrument['phase'] = sc.array(dims=['chopper'],
                                   values=[phase.value],
                                   unit=sc.units.rad)

    # window opening angle
    instrument['frame_start'] = sc.array(
        dims=['frame'], values=[0], unit=sc.units.rad) * sc.ones(
            dims=['chopper'], shape=[1])
    # window closing angle
    instrument['frame_end'] = sc.array(
        dims=['frame'], values=[window_size], unit=sc.units.rad) * sc.ones(
            dims=['chopper'], shape=[1])

    instrument['tdc'] = sc.zeros(dims=['chopper'], shape=[
        1
    ]) if tdc is None else sc.array(dims=['chopper'], values=[tdc])
    instrument['choppers'] = sc.array(dims=['chopper'], values=['chopper'])
    return instrument


def test_frames_analytical_one_chopper_one_cutout():
    offset = 5.0 * sc.units.us  # start of pulse t offset from 0
    window_opening_t = 5.0 * sc.units.us
    pulse_length = 10.0 * sc.units.us
    instrument = _single_chopper_beamline(window_opening_t, pulse_length)
    frames = frames_analytical(instrument, offset=offset)
    # Following results best understood sketching simple time-distance
    # diagram of above.
    assert allclose(
        frames['left_edges'].data,
        sc.array(dims=['frame'], values=[-5.0], unit=sc.units.us) + offset)
    assert allclose(
        frames['right_edges'].data,
        sc.array(dims=['frame'], values=[15.0], unit=sc.units.us) + offset)
    assert allclose(
        frames['shifts'].data,
        sc.array(dims=['frame'], values=[5.0], unit=sc.units.us) + offset)


def test_frame_analytical_one_chopper_one_cutout_different_pulse_offset():
    # Offsetting the pulse start time should impact directly on the
    # frame edges and shifts
    window_opening_t = 5.0 * sc.units.us
    pulse_length = 10.0 * sc.units.us
    for offset in [0.0 * sc.units.us, 5.0 * sc.units.us]:
        instrument = _single_chopper_beamline(window_opening_t, pulse_length)
        frames = frames_analytical(instrument, offset=offset)

        # Following results best understood sketching simple time-distance
        # diagram of above.
        assert allclose(
            frames['left_edges'].data,
            sc.array(dims=['frame'], values=[-5.0], unit=sc.units.us) + offset)
        assert allclose(
            frames['right_edges'].data,
            sc.array(dims=['frame'], values=[15.0], unit=sc.units.us) + offset)
        assert allclose(
            frames['shifts'].data,
            sc.array(dims=['frame'], values=[5.0], unit=sc.units.us) + offset)


def test_frames_analytical_with_different_window_times():
    # Setting different window times will NOT
    # change the shift (windows always centered on center of pulse),
    # but this will change the frame edges
    offset = 0.0 * sc.units.us  # start of pulse t offset from 0
    pulse_length = 10.0 * sc.units.us
    invariant_shift = (pulse_length / 2) + offset

    window_opening_t = 10.0 * sc.units.us
    instrument = _single_chopper_beamline(window_opening_t, pulse_length)
    frames = frames_analytical(instrument, offset=offset)
    assert allclose(
        frames['left_edges'].data,
        sc.array(dims=['frame'], values=[-10.0], unit=sc.units.us) + offset)
    assert allclose(
        frames['right_edges'].data,
        sc.array(dims=['frame'], values=[20.0], unit=sc.units.us) + offset)
    assert allclose(
        frames['shifts'].data,
        sc.array(dims=['frame'],
                 values=[invariant_shift.value],
                 unit=sc.units.us))

    window_opening_t = 2.0 * sc.units.us
    instrument = _single_chopper_beamline(window_opening_t, pulse_length)
    frames = frames_analytical(instrument, offset=offset)
    assert allclose(
        frames['left_edges'].data,
        sc.array(dims=['frame'], values=[-2.0], unit=sc.units.us) + offset)
    assert allclose(
        frames['right_edges'].data,
        sc.array(dims=['frame'], values=[12.0], unit=sc.units.us) + offset)
    assert allclose(
        frames['shifts'].data,
        sc.array(dims=['frame'],
                 values=[invariant_shift.value],
                 unit=sc.units.us))
