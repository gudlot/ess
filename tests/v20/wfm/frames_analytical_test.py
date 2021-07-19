from ess.v20.wfm.frames_analytical import frames_analytical
import scipp as sc
import numpy as np
import itertools
from .common import _chopper_ang_freq, allclose


def _common_beamline(pulse_length, chopper_z_offsets=[5.0]):
    instrument = sc.Dataset()
    # single pixel set 10m down the beam
    instrument['position'] = sc.vector(value=[0, 0, 10], unit=sc.units.m)
    # Describe choppers offset down the beam
    z_offset = sc.array(dims=['chopper'],
                        values=chopper_z_offsets,
                        unit=sc.units.m)
    no_offset = sc.zeros(dims=['chopper'],
                         shape=[len(chopper_z_offsets)],
                         unit=sc.units.m)
    instrument['distance'] = sc.geometry.position(no_offset, no_offset,
                                                  z_offset)
    instrument['pulse_length'] = pulse_length
    return instrument


def _multi_chopper_single_cutout_beamline(window_opening_t,
                                          pulse_length,
                                          phases,
                                          chopper_z_offsets,
                                          window_size=np.pi / 4 *
                                          sc.units.rad):
    instrument = _common_beamline(chopper_z_offsets=chopper_z_offsets,
                                  pulse_length=pulse_length)
    # We now set out to engineer a single cutout to deliver a specified
    # length sub-pulse.
    window_opening_t = sc.to_unit(window_opening_t, sc.units.s)
    chopper_ang_frequency = _chopper_ang_freq(window_opening_t, window_size)
    instrument['angular_frequency'] = sc.broadcast(
        chopper_ang_frequency,
        dims=['chopper'],
        shape=[len(chopper_z_offsets)])

    instrument['phase'] = phases

    # window opening angle
    instrument['frame_start'] = sc.array(
        dims=['frame'], values=[0], unit=sc.units.rad) * sc.ones(
            dims=['chopper'], shape=[len(chopper_z_offsets)])
    # window closing angle
    instrument['frame_end'] = sc.ones(dims=['frame'], shape=[1]) * sc.ones(
        dims=['chopper'], shape=[len(chopper_z_offsets)]) * window_size

    instrument['choppers'] = sc.array(
        dims=['chopper'],
        values=['chopper' + str(i) for i in range(len(chopper_z_offsets))])
    return instrument


# Single chopper single cutout
def _single_chopper_beamline(window_opening_t,
                             pulse_length,
                             window_size=np.pi / 4 * sc.units.rad,
                             phase=None):

    instrument = _common_beamline(chopper_z_offsets=[5.0],
                                  pulse_length=pulse_length)
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


def _single_chopper_double_window(window_opening_t,
                                  pulse_length,
                                  window_size=np.pi / 4 * sc.units.rad):

    instrument = _common_beamline(chopper_z_offsets=[5.0],
                                  pulse_length=pulse_length)
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
    for offset in [0.0 * sc.units.us, 1.0 * sc.units.us, 2.0 * sc.units.us]:
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


def test_frames_analytical_dual_chopper_single_cutout_():
    # Test double chopper blind setup.
    offset = 0.0 * sc.units.us  # start of pulse t offset from 0
    window_opening_t = 1.0 * sc.units.us
    window_size = np.pi / 8.0 * sc.units.rad
    pulse_length = 5.0 * sc.units.us
    freq = 2 * np.pi * (sc.units.rad / sc.units.s)

    freq = _chopper_ang_freq(window_opening_t, window_size)
    pulse_end = pulse_length + offset
    # Assemble our time-edges for chopper openings closings
    # that is 2 choppers * 1 cutout * 2 edges per cutout
    # (no overlap) or gaps between edges
    l_time_edges = [pulse_end + (window_opening_t * i) for i in range(2)]
    r_time_edges = [window_opening_t + l_edge for l_edge in l_time_edges]
    # Get left and right edges as interleaved
    time_edges = list(
        itertools.chain(*[p for p in zip(l_time_edges, r_time_edges)]))
    # This gives a contiguous time split into pulse, ch1 cutout1, ch2 cutout1
    # Find cutout centre time
    time_centers = [(time_edges[i] + time_edges[i + 1]) * 0.5
                    for i in range(0, len(time_edges), 2)]
    # Convert this to phases for each chopper cutout
    phases = sc.array(dims=['chopper'],
                      values=[(t * freq).value for t in time_centers],
                      unit=sc.units.rad)
    # Now assemble choppers
    instrument = _multi_chopper_single_cutout_beamline(
        window_opening_t=window_opening_t,
        window_size=window_size,
        pulse_length=pulse_length,
        phases=phases,
        chopper_z_offsets=[4.5, 5.5])
    frames = frames_analytical(instrument, plot=False)
    # Chopper 2 provides maximum and minium slope constraints and
    # has time center at 6.5 (see time_centers) which sets shift
    assert sc.isclose(frames['shifts'].data['frame', 0], time_centers[1]).value
