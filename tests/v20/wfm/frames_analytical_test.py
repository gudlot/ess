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

    instrument['phase'] = phase if phase.shape[0] > 0 else sc.broadcast(
        phase, dims=['chopper'], shape=[1])

    # window opening angle
    instrument['frame_start'] = sc.array(
        dims=['frame'], values=[0], unit=sc.units.rad) * sc.ones(
            dims=['chopper'], shape=[1])
    # window closing angle
    instrument['frame_end'] = sc.ones(dims=['frame'], shape=[1]) * sc.ones(
        dims=['chopper'], shape=[1]) * window_size

    instrument['choppers'] = sc.array(dims=['chopper'], values=['chopper'])
    return instrument


def _make_chopper_phases(pulse_length, window_opening_t, window_size,
                         start_chopper_padding_t, n_choppers):
    freq = _chopper_ang_freq(window_opening_t, window_size)
    # After pulse, get t-gap then contiguous
    # times from sequentially opening choppers
    l_time_edges = [
        pulse_length + start_chopper_padding_t + (window_opening_t * i)
        for i in range(n_choppers)
    ]
    # Convert this to phases for each chopper cutout
    phases = sc.array(dims=['chopper'],
                      values=[(t * freq).value for t in l_time_edges],
                      unit=sc.units.rad)
    return phases


def _test_single_chopper_single_cutout(offset, padding, pulse_length,
                                       window_opening_t, window_size):
    phase = _make_chopper_phases(pulse_length=pulse_length,
                                 window_opening_t=window_opening_t,
                                 window_size=window_size,
                                 start_chopper_padding_t=padding,
                                 n_choppers=1)
    source_to_chopper_distance = 5.0 * sc.units.m
    source_to_pixel_distance = 10.0 * sc.units.m
    max_gradient = source_to_chopper_distance / (
        padding)  # 5.0 is source - chopper distance
    l_edge_predicted = (source_to_pixel_distance /
                        max_gradient) + (offset + pulse_length)
    min_gradient = source_to_pixel_distance / (pulse_length + padding +
                                               window_opening_t)
    r_edge_predicted = (source_to_pixel_distance / min_gradient) + (
        offset + pulse_length + padding + window_opening_t)
    shift_predicted = window_opening_t / 2.0 + offset + pulse_length + padding
    instrument = _single_chopper_beamline(window_opening_t=window_opening_t,
                                          pulse_length=pulse_length,
                                          window_size=window_size,
                                          phase=phase)
    frames = frames_analytical(instrument, offset=offset)
    # Following results best understood sketching simple time-distance
    # diagram of above.
    assert allclose(
        frames['left_edges'].data,
        sc.broadcast(dims=['frame'], shape=[1], x=l_edge_predicted))
    assert allclose(
        frames['right_edges'].data,
        sc.broadcast(dims=['frame'], shape=[1], x=r_edge_predicted))
    assert allclose(frames['shifts'].data,
                    sc.broadcast(dims=['frame'], shape=[1], x=shift_predicted))
    return frames


def test_frames_analytical_one_chopper_one_cutout():
    offset = 0.0 * sc.units.us  # start of pulse t offset from 0
    window_opening_t = 0.5 * sc.units.us
    pulse_length = 10.0 * sc.units.us
    window_size = (np.pi / 4.0) * sc.units.rad  # 45 degree opening

    padding = 1.0 * sc.units.us
    _test_single_chopper_single_cutout(offset, padding, pulse_length,
                                       window_opening_t, window_size)


def test_frames_analytical_one_chopper_one_cutout_different_pulse_offset():
    # Offsetting the pulse start time should impact directly on the
    # frame edges and shifts
    offset = 5.0 * sc.units.us  # start of pulse t offset from 0
    window_opening_t = 0.5 * sc.units.us
    pulse_length = 10.0 * sc.units.us
    window_size = (np.pi / 4.0) * sc.units.rad  # 45 degree opening

    padding = 1.0 * sc.units.us
    a = _test_single_chopper_single_cutout(offset, padding, pulse_length,
                                           window_opening_t, window_size)
    # Now with new larger offset
    offset += 1.0 * sc.units.us
    b = _test_single_chopper_single_cutout(offset, padding, pulse_length,
                                           window_opening_t, window_size)
    assert sc.all(b['left_edges'].data > a['left_edges'].data).value
    assert sc.all(b['right_edges'].data > a['right_edges'].data).value
    assert sc.all(b['shifts'].data > a['shifts'].data).value
    assert allclose(b['shifts'].data, a['shifts'].data + 1.0 * sc.units.us)


def test_frames_analytical_with_different_window_times():
    offset = 5.0 * sc.units.us  # start of pulse t offset from 0
    window_opening_t = 0.5 * sc.units.us
    pulse_length = 10.0 * sc.units.us
    window_size = (np.pi / 4.0) * sc.units.rad  # 45 degree opening

    padding = 1.0 * sc.units.us
    a = _test_single_chopper_single_cutout(offset, padding, pulse_length,
                                           window_opening_t, window_size)
    # increase opening times on second pass
    b = _test_single_chopper_single_cutout(offset, padding, pulse_length,
                                           window_opening_t + window_opening_t,
                                           window_size)
    assert allclose(b['shifts'].data,
                    a['shifts'].data + (window_opening_t / 2.0))


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
