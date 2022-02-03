# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

from typing import Tuple
import scipp as sc
from .common import gravity_vector
from . import conversions
from . import normalization
from scipp.interpolate import interp1d


def _make_coordinate_transform_graphs(gravity: bool) -> Tuple[dict, dict]:
    """
    Create unit conversion graphs.
    The gravity parameter can be used to turn on or off the effects of gravity.
    """
    data_graph = conversions.sans_elastic(gravity=gravity)
    monitor_graph = conversions.sans_monitor()
    return data_graph, monitor_graph


def _convert_to_wavelength(data: sc.DataArray, monitors: dict, data_graph: dict,
                           monitor_graph: dict) -> Tuple[sc.DataArray, dict]:
    """
    Convert the data array and all the items inside the dict of monitors to wavelength
    using a pre-defined conversion graph.
    """
    data = data.transform_coords("wavelength", graph=data_graph)
    monitors = {
        key: monitors[key].transform_coords("wavelength", graph=monitor_graph)
        for key in monitors
    }
    return data, monitors


def _denoise_and_rebin_monitors(monitors: dict, wavelength_bins: sc.Variable,
                                non_background_range: sc.Variable) -> dict:
    """
    Subtract a background baseline from monitor counts, taken as the mean of the counts
    outside the specified ``non_background_range``.
    """
    return {
        key: normalization.subtract_background_and_rebin(
            monitors[key],
            wavelength_bins=wavelength_bins,
            non_background_range=non_background_range)
        for key in monitors
    }


def _direct_beam(direct_beam: sc.DataArray,
                 wavelength_bins: sc.Variable) -> sc.DataArray:
    """
    If the wavelength binning of the direct beam function does not match the requested
    ``wavelength_bins``, perform a 1d interpolation of the function onto the bins.
    """
    if sc.identical(direct_beam.coords['wavelength'], wavelength_bins):
        return direct_beam
    func = interp1d(sc.values(direct_beam), 'wavelength')
    direct_beam = func(wavelength_bins, midpoints=True)
    logger = sc.get_logger()
    logger.warning('An interpolation was performed on the direct_beam function. '
                   'The variances in the direct_beam function have been dropped.')
    return direct_beam


def _denominator(direct_beam: sc.DataArray, data_incident_monitor: sc.DataArray,
                 transmission_fraction: sc.DataArray,
                 solid_angle: sc.Variable) -> sc.DataArray:
    """
    Compute the denominator term.
    Because we are histogramming the Q values of the denominator further down in the
    workflow, we convert the wavelength coordinate of the denominator from bin edges to
    bin centers.
    """
    denominator = (solid_angle * direct_beam * data_incident_monitor *
                   transmission_fraction)
    # TODO: once scipp-0.12 is released, use sc.midpoints()
    denominator.coords['wavelength'] = 0.5 * (denominator.coords['wavelength'][1:] +
                                              denominator.coords['wavelength'][:-1])
    return denominator


def _convert_to_q_and_merge_spectra(data: sc.DataArray, graph: dict,
                                    wavelength_bands: sc.Variable, q_bins: sc.Variable,
                                    gravity: bool) -> sc.DataArray:
    """
    Convert the data to momentum vector Q. This accepts both dense and event data.
    The final step merges all spectra:
      - In the case of event data, events in all bins are concatenated
      - In the case of dense data, counts in all spectra are summed
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

    data_graph, monitor_graph = _make_coordinate_transform_graphs(gravity=gravity)

    data, monitors = _convert_to_wavelength(data=data,
                                            monitors=monitors,
                                            data_graph=data_graph,
                                            monitor_graph=monitor_graph)

    monitors = _denoise_and_rebin_monitors(
        monitors=monitors,
        wavelength_bins=wavelength_bins,
        non_background_range=monitor_non_background_range)

    transmission_fraction = normalization.transmission_fraction(**monitors)

    solid_angle = normalization.solid_angle_of_rectangular_pixels(
        data,
        pixel_width=data.coords['pixel_width'],
        pixel_height=data.coords['pixel_height'])

    direct_beam = _direct_beam(direct_beam=direct_beam, wavelength_bins=wavelength_bins)

    denominator = _denominator(direct_beam=direct_beam,
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

    data_q = _convert_to_q_and_merge_spectra(data=data,
                                             graph=data_graph,
                                             wavelength_bands=wavelength_bands,
                                             q_bins=q_bins,
                                             gravity=gravity)

    denominator_q = _convert_to_q_and_merge_spectra(data=denominator,
                                                    graph=data_graph,
                                                    wavelength_bands=wavelength_bands,
                                                    q_bins=q_bins,
                                                    gravity=gravity)

    normalized = normalization.normalize(numerator=data_q,
                                         denominator=denominator_q,
                                         dim='Q')

    return normalized
