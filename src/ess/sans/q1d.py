# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import g
import scippneutron as scn
from . import conversions
from . import normalization
from ..wfm.tools import to_bin_centers, to_bin_edges
from scipp.interpolate import interp1d


def q1d(data,
        data_incident_monitor,
        data_transmission_monitor,
        direct_incident_monitor,
        direct_transmission_monitor,
        direct_beam,
        wavelength_bins,
        q_bins,
        number_of_wavelength_bands=1,
        monitor_background_threshold=sc.scalar(30.0, unit='counts'),
        histogram_output=True):

    logger = sc.get_logger()

    # 1. Convert to wavelength =========================================================

    # Make shallow copy of data so as to not add coords in-place
    data = data.copy(deep=False)

    monitors = {
        'data_incident_monitor': data_incident_monitor,
        'data_transmission_monitor': data_transmission_monitor,
        'direct_incident_monitor': direct_incident_monitor,
        'direct_transmission_monitor': direct_transmission_monitor
    }

    # Create unit conversion graphs
    graph = conversions.sans_elastic()
    graph_monitor = conversions.sans_monitor()

    # Add gravity coordinate
    data.coords["gravity"] = sc.vector(value=[0, -1, 0]) * g

    # Convert to wavelength and bin to requested range
    data = data.transform_coords("wavelength", graph=graph)
    data = sc.bin(data,
                  edges=[
                      sc.concat([wavelength_bins.min(),
                                 wavelength_bins.max()],
                                dim='wavelength')
                  ])
    monitors_wav = {
        key: monitors[key].transform_coords("wavelength", graph=graph_monitor)
        for key in monitors
    }

    # Align monitors to common wavelength axis
    for key in monitors_wav:
        monitors_wav[key] = normalization.subtract_background_and_rebin(
            monitors_wav[key],
            wavelength_bins=wavelength_bins,
            threshold=monitor_background_threshold)

    # 2. Compute denominator ===========================================================

    # Transmission fraction
    transmission_fraction = normalization.transmission_fraction(**monitors_wav)

    # Solid angle
    solid_angle = normalization.solid_angle(data,
                                            pixel_width=data.coords['pixel_width'],
                                            pixel_height=data.coords['pixel_height'])

    # Interpolate the direct beam function to the requested wavelength binning
    interpolate_direct_beam = True
    if direct_beam.coords['wavelength'].sizes['wavelength'] > direct_beam.sizes[
            'wavelength']:
        if sc.identical(direct_beam.coords['wavelength'], wavelength_bins):
            interpolate_direct_beam = False
    else:
        if sc.identical(
                to_bin_edges(direct_beam.coords['wavelength'], dim='wavelength'),
                wavelength_bins):
            interpolate_direct_beam = False
    if interpolate_direct_beam:
        func = interp1d(sc.values(direct_beam), 'wavelength')
        direct_beam = func(wavelength_bins, midpoints=True)
        logger.warning('ess.sans.q1d: An interpolation was performed on the '
                       'direct_beam function. The variances in the direct_beam '
                       'function have been ignored.')

    # Compute denominator and copy coords needed for conversion to Q
    denominator = (solid_angle * direct_beam * monitors_wav['data_incident_monitor'] *
                   transmission_fraction)
    denominator.coords['position'] = data.meta['position']
    denominator.coords['gravity'] = data.meta['gravity']
    denominator.coords['sample_position'] = data.meta['sample_position']
    denominator.coords['source_position'] = data.meta['source_position']
    denominator.coords['wavelength'] = to_bin_centers(denominator.coords['wavelength'],
                                                      dim='wavelength')

    # 3. Make wavelength bands if necessary ============================================

    if number_of_wavelength_bands > 1:
        if wavelength_bins.sizes['wavelength'] % number_of_wavelength_bands != 0:
            logger.warning('ess.sans.q1d: The number of wavelength bins is not an '
                           'integer multiple of the number of wavelength bands.')
        data = sc.fold(data,
                       dim='wavelength',
                       sizes={
                           'band': number_of_wavelength_bands,
                           'wavelength': 1
                       })
        denominator = sc.fold(denominator,
                              dim='wavelength',
                              sizes={
                                  'band':
                                  number_of_wavelength_bands,
                                  'wavelength':
                                  wavelength_bins.sizes['wavelength'] //
                                  number_of_wavelength_bands
                              })

    # 4. Convert from wavelength to Q ==================================================

    data = data.transform_coords("Q", graph=graph)
    q_boundaries = sc.concat([q_bins.min(), q_bins.max()], dim='Q')
    q_binned = sc.bin(data, edges=[q_boundaries])
    q_summed = q_binned.bins.concat('spectrum')

    den_q = denominator.transform_coords("Q", graph=graph)
    den_hist = sc.histogram(den_q, bins=q_bins)
    den_q_summed = den_hist.sum('spectrum')

    # 5. Normalize =====================================================================

    normalized = q_summed.bins / sc.lookup(func=den_q_summed, dim='Q')
    if histogram_output:
        normalized = sc.histogram(normalized, bins=q_bins)
    return normalized
