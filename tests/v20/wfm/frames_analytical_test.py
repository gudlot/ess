from ess.v20.wfm.frames_analytical import frames_analytical
import scipp as sc
import numpy as np
from .common import _single_chopper_beamline, _chopper_ang_freq, allclose

def _single_chopper_double_window(window_opening_t,
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

    blind_t = sc.to_unit(pulse_length, sc.units.s) - window_opening_t
    # For calc simplicity to center cutout opening over center of pulse
    phase_offset_t = blind_t / 2
    phase = phase_offset_t * chopper_ang_frequency
    instrument['phase'] = sc.array(dims=['chopper'],
                                   values=[phase.value],
                                   unit=sc.units.rad)

    # window opening angle
    instrument['frame_start'] = sc.array(
        dims=['frame'], values=[0, np.pi], unit=sc.units.rad) * sc.ones(
            dims=['chopper'], shape=[1])
    # window closing angle
    instrument['frame_end'] = sc.concatenate(
        dim='frame', x=window_size, y=window_size +
        np.pi * sc.units.rad) * sc.ones(dims=['chopper'], shape=[1])

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


def test_frames_analytical_one_chopper_one_cutout_different_pulse_offset():
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


def test_frames_analytical_one_chopper_one_cutout_shift_phase():
    offset = 0.0 * sc.units.us  # start of pulse t offset from 0
    window_opening_t = 5.0 * sc.units.us
    window_size = np.pi / 4.0 * sc.units.rad
    pulse_length = 10.0 * sc.units.us
    freq = _chopper_ang_freq(window_opening_t, window_size)
    for phase in [i * sc.units.rad for i in [0.0, np.pi / 4.0, np.pi]]:
        tshift = phase / freq  # Expected tshift from chopper phase
        instrument = _single_chopper_beamline(window_opening_t,
                                              pulse_length,
                                              window_size=window_size,
                                              phase=phase)
        frames = frames_analytical(instrument, offset=offset)
        # Factor 2 comes from the geometry. Pixel 2 * source - chopper distance
        assert allclose(
            frames['left_edges'].data,
            sc.array(dims=['frame'], values=[-10.0], unit=sc.units.us) +
            offset + (tshift * 2))
        assert allclose(
            frames['right_edges'].data,
            sc.array(dims=['frame'], values=[10.0], unit=sc.units.us) +
            offset + (tshift * 2))


def test_frames_analytical_one_chopper_dual_cutout():
    window_opening_t = 5.0 * sc.units.us
    window_size = np.pi / 4.0 * sc.units.rad
    pulse_length = 10.0 * sc.units.us
    freq = _chopper_ang_freq(window_opening_t, window_size)
    # windows are opposite (pi apart)
    time_between_frames = np.pi * sc.units.rad / freq
    instrument = _single_chopper_double_window(window_opening_t, pulse_length,
                                               window_size)
    frames = frames_analytical(instrument)
    np.isclose(frames['left_edges'].data.values[0] + time_between_frames.value,
               frames['left_edges'].data.values[1])
    np.isclose(
        frames['right_edges'].data.values[0] + time_between_frames.value,
        frames['right_edges'].data.values[1])
