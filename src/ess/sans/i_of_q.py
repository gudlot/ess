# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from typing import Tuple
import scipp as sc
from .common import gravity_vector
from . import conversions
from . import normalization
from scipp.interpolate import interp1d


def make_coordinate_transform_graphs(gravity: bool) -> Tuple[dict, dict]:
    """
    Create unit conversion graphs.
    The gravity parameter can be used to turn on or off the effects of gravity.

    :param gravity: If True, the coordinate transformation graph will incorporate the
        effects of the Earth's gravitational field on the flight path of the neutrons
        when computing the scattering angle.
    """
    data_graph = conversions.sans_elastic(gravity=gravity)
    monitor_graph = conversions.sans_monitor()
    return data_graph, monitor_graph


def convert_to_wavelength(data: sc.DataArray, monitors: dict, data_graph: dict,
                          monitor_graph: dict) -> Tuple[sc.DataArray, dict]:
    """
    Convert the data array and all the items inside the dict of monitors to wavelength
    using a pre-defined conversion graph.

    :param data: The data from the measurement that is to be converted to wavelength.
    :param monitors: A dict of monitors. All entries in the dict will be converted to
        wavelength.
    :param data_graph: The coordinate transformation graph to be used for the data.
    :param monitor_graph: The coordinate transformation graph to be used for the
        monitors.
    """
    data = data.transform_coords("wavelength", graph=data_graph)
    monitors = {
        key: monitors[key].transform_coords("wavelength", graph=monitor_graph)
        for key in monitors
    }
    return data, monitors


def denoise_and_rebin_monitors(monitors: Union[dict, sc.DataArray],
                               wavelength_bins: sc.Variable,
                               non_background_range: sc.Variable = None) -> dict:
    """
    Subtract a background baseline from monitor counts, taken as the mean of the counts
    outside the specified ``non_background_range``.

    :param monitors: A DataArray containing monitor data, or a dict of monitor
        DataArrays. In the case of a dict of monitors, all entries in the dict will
        be background subtracted and rebinned.
    :param wavelength_bins: The binning in wavelength to use for the rebinning.
    :param non_background_range: The range of wavelengths that defines the data which
        does not constitute background. Everything outside this range is treated as
        background counts.
    """
    if isinstance(monitors, dict):
        return {
            key:
            _subtract_background_and_rebin(monitors[key],
                                           wavelength_bins=wavelength_bins,
                                           non_background_range=non_background_range)
            for key in monitors
        }
    else:
        return _subtract_background_and_rebin(monitors,
                                              wavelength_bins=wavelength_bins,
                                              non_background_range=non_background_range)


def _subtract_background_and_rebin(
        data: sc.DataArray,
        wavelength_bins: sc.Variable,
        non_background_range: sc.Variable = None) -> sc.DataArray:
    """
    Subtracts background value from data counts and performs a wavelength rebin.
    The background is computed as the mean value of all the counts outside of the given
    ``non_background_range``.

    :param data: The DataArray containing the monitor data to be de-noised and rebinned.
    :param wavelength_bins: The wavelength binning to apply when rebinning the data.
    :param non_background_range: The range of wavelengths that defines the data which
        does not constitute background. Everything outside this range is treated as
        background counts.
    """
    if non_background_range is not None:
        dim = non_background_range.dim
        below = data[dim, :non_background_range[0]]
        above = data[dim, non_background_range[1]:]
        # TODO: if we implement `ones_like` for data arrays, we could use that here
        # instead of dividing the below and above pieces by themselves
        divisor = sc.nansum(below / below).data + sc.nansum(above / above).data
        background = (below.sum().data + above.sum().data) / divisor
        data = data - background
    return sc.rebin(data, "wavelength", wavelength_bins)


def resample_direct_beam(direct_beam: sc.DataArray,
                         wavelength_bins: sc.Variable) -> sc.DataArray:
    """
    If the wavelength binning of the direct beam function does not match the requested
    ``wavelength_bins``, perform a 1d interpolation of the function onto the bins.

    :param direct_beam: The DataArray containing the direct beam function (it should
        have a dimension of wavelength).
    :param wavelength_bins: The binning in wavelength that the direct beam function
        should be resampled to.
    """
    if sc.identical(direct_beam.coords['wavelength'], wavelength_bins):
        return direct_beam
    func = interp1d(sc.values(direct_beam), 'wavelength')
    direct_beam = func(wavelength_bins, midpoints=True)
    logger = sc.get_logger()
    logger.warning('An interpolation was performed on the direct_beam function. '
                   'The variances in the direct_beam function have been dropped.')
    return direct_beam


def convert_to_q_and_merge_spectra(data: sc.DataArray, graph: dict,
                                   wavelength_bands: sc.Variable, q_bins: sc.Variable,
                                   gravity: bool) -> sc.DataArray:
    """
    Convert the data to momentum vector Q. This accepts both dense and event data.
    The final step merges all spectra:
      - In the case of event data, events in all bins are concatenated
      - In the case of dense data, counts in all spectra are summed

    :param data: The DataArray containing the data that is to be converted to Q.
    :param graph: The coordinate conversion graph used to perform the conversion to Q.
    :param wavelength_bands: Defines bands in wavelength that can be used to separate
        different wavelength ranges that contribute to different regions in Q space.
        Note that this needs to be defined, so if all wavelengths should be used, this
        should simply be a start and end edges that encompass the entire wavelength
        range.
    :param q_bins: The binning in Q to be used.
    :param gravity: If True, include the effects of gravity when computing the
        scattering angle.
    """
    if gravity:
        data = data.copy(deep=False)
        data.coords["gravity"] = gravity_vector()

    if data.bins is not None:
        out = _convert_events_to_q_and_merge_spectra(data=data,
                                                     graph=graph,
                                                     q_bins=q_bins,
                                                     wavelength_bands=wavelength_bands)
    else:
        out = _convert_dense_to_q_and_merge_spectra(data=data,
                                                    graph=graph,
                                                    q_bins=q_bins,
                                                    wavelength_bands=wavelength_bands)
    if wavelength_bands.sizes['wavelength'] == 2:
        out = out['wavelength', 0]
    return out


def _convert_events_to_q_and_merge_spectra(
        data: sc.DataArray, graph: dict, q_bins: sc.Variable,
        wavelength_bands: sc.Variable) -> sc.DataArray:
    """
    Convert event data to momentum vector Q.
    """
    data_q = data.transform_coords("Q", graph=graph)

    # TODO: once scipp-0.12 is out, we no longer need to move the attr into the coords
    data_q.bins.coords['wavelength'] = data_q.bins.attrs.pop('wavelength')

    q_summed = data_q.bins.concat('spectrum')
    return sc.bin(q_summed, edges=[wavelength_bands, q_bins])


def _convert_dense_to_q_and_merge_spectra(
        data: sc.DataArray, graph: dict, q_bins: sc.Variable,
        wavelength_bands: sc.Variable) -> sc.DataArray:
    """
    Convert dense data to momentum vector Q.
    """
    bands = []
    data_q = data.transform_coords("Q", graph=graph)
    data_q.coords['wavelength'] = data_q.attrs.pop('wavelength')
    for i in range(wavelength_bands.sizes['wavelength'] - 1):
        band = data_q['wavelength', wavelength_bands[i]:wavelength_bands[i + 1]]
        bands.append(sc.histogram(band, bins=q_bins).sum('spectrum'))
    q_summed = sc.concat(bands, 'wavelength')
    return q_summed


def to_I_of_Q(data: sc.DataArray,
              data_incident_monitor: sc.DataArray,
              data_transmission_monitor: sc.DataArray,
              direct_incident_monitor: sc.DataArray,
              direct_transmission_monitor: sc.DataArray,
              direct_beam: sc.DataArray,
              wavelength_bins: sc.Variable,
              q_bins: sc.Variable,
              gravity: bool = False,
              monitor_non_background_range: sc.Variable = None,
              wavelength_bands: sc.Variable = None) -> sc.DataArray:
    """
    Compute the scattering cross-section I(Q) for a SANS experimental run, performing
    binning in Q and a normalization based on monitor data and a direct beam function.

    The main steps of the workflow are:

       * Generate a coordinate transformation graph from ``tof`` to ``Q``, that also
         includes ``wavelength``.
       * Convert the detector data and monitors to wavelength.
       * Remove the background noise from the monitors and align them to a common
         binning axis.
       * Compute the transmission fraction from the monitor data.
       * Compute the solid angles of the detector pixels.
       * Resample the direct beam function to the same wavelength binning as the
         monitors.
       * Combine solid angle, direct beam, transmission fraction and incident monitor
         counts to compute the denominator for the normalization.
       * Convert the detector data to momentum vector Q.
       * Convert the denominator to momentum vector Q.
       * Normalize the detector data.

    :param data: The DataArray containing the detector data. This can be both events
        or dense (histogrammed) data.
    :param data_incident_monitor: The histogrammed counts of the incident monitor
        during the measurement (sample or background) run, as a function of wavelength.
    :param data_transmission_monitor:  The histogrammed counts of the transmission
        monitor during the measurement (sample or background) run, as a function of
        wavelength.
    :param direct_incident_monitor: The histogrammed counts of the incident monitor
        during the direct (empty sample holder) run, as a function of wavelength.
    :param direct_transmission_monitor: The histogrammed counts of the transmission
        monitor during the direct (empty sample holder) run, as a function of
        wavelength.
    :param direct_beam: The direct beam function of the instrument (histogrammed,
        depends on wavelength).
    :param wavelength_bins: The binning in the wavelength dimension to be used.
    :param q_bins: The binning in the Q dimension to be used.
    :param gravity: Include the effects of gravity when computing the scattering angle
        if True.
    :param monitor_non_background_range: The range of wavelengths for the monitors that
        are considered to not be part of the background. This is used to compute the
        background level on each monitor, which then gets subtracted from each monitor's
        counts.
    :param wavelength_bands: If defined, return the data as a set of bands in the
        wavelength dimension. This is useful for separating different wavelength ranges
        that contribute to different regions in Q space.
    """

    monitors = {
        'data_incident_monitor': data_incident_monitor,
        'data_transmission_monitor': data_transmission_monitor,
        'direct_incident_monitor': direct_incident_monitor,
        'direct_transmission_monitor': direct_transmission_monitor
    }

    data_graph, monitor_graph = make_coordinate_transform_graphs(gravity=gravity)

    data, monitors = convert_to_wavelength(data=data,
                                           monitors=monitors,
                                           data_graph=data_graph,
                                           monitor_graph=monitor_graph)

    monitors = denoise_and_rebin_monitors(
        monitors=monitors,
        wavelength_bins=wavelength_bins,
        non_background_range=monitor_non_background_range)

    transmission_fraction = normalization.transmission_fraction(**monitors)

    direct_beam = resample_direct_beam(direct_beam=direct_beam,
                                       wavelength_bins=wavelength_bins)

    solid_angle = normalization.solid_angle_of_rectangular_pixels(
        data,
        pixel_width=data.coords['pixel_width'],
        pixel_height=data.coords['pixel_height'])

    denominator = normalization.compute_denominator(
        direct_beam=direct_beam,
        data_incident_monitor=monitors['data_incident_monitor'],
        transmission_fraction=transmission_fraction,
        solid_angle=solid_angle)
    # Insert a copy of coords needed for conversion to Q.
    # TODO: can this be avoided by copying the Q coords from the converted numerator?
    for coord in ['position', 'sample_position', 'source_position']:
        denominator.coords[coord] = data.meta[coord]

    # In the case where no wavelength bands are requested, we create a single wavelength
    # band to make sure we select the correct wavelength range that corresponds to
    # wavelength_bins
    if wavelength_bands is None:
        wavelength_bands = sc.concat(
            [wavelength_bins.min(), wavelength_bins.max()], dim='wavelength')

    data_q = convert_to_q_and_merge_spectra(data=data,
                                            graph=data_graph,
                                            wavelength_bands=wavelength_bands,
                                            q_bins=q_bins,
                                            gravity=gravity)

    denominator_q = convert_to_q_and_merge_spectra(data=denominator,
                                                   graph=data_graph,
                                                   wavelength_bands=wavelength_bands,
                                                   q_bins=q_bins,
                                                   gravity=gravity)

    normalized = normalization.normalize(numerator=data_q,
                                         denominator=denominator_q,
                                         dim='Q')

    return normalized
