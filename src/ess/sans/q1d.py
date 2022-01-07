# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
from scipp.constants import g
import scippneutron as scn
from . import conversions
from . import normalization


def q1d(data,
        data_incident_monitor,
        data_transmission_monitor,
        direct_incident_monitor,
        direct_transmission_monitor,
        detector_efficiency,
        wavelength_bins,
        q_bins,
        monitor_background_threshold=sc.scalar(30.0, unit='counts')):

    # 1. Convert to wavelength =========================================================

    # Make shallow copy of data so as to not add coords in-place
    data = data.copy(deep=False)

    monitors = {
        'data_incident': data_incident_monitor,
        'data_transmission': data_transmission_monitor,
        'direct_incident': direct_incident_monitor,
        'direct_transmission': direct_transmission_monitor
    }

    # Create unit conversion graphs
    graph = conversions.sans_elastic()
    graph_monitor = conversions.sans_monitor()

    # Add gravity coordinate
    data.coords["gravity"] = sc.vector(value=[0, -1, 0]) * g

    # Convert to wavelength
    data = data.transform_coords("wavelength", graph=graph)
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

    # Compute denominator and copy coords needed for conversion to Q
    denominator = solid_angle * (
        detector_efficiency *
        (monitors_wav['data_incident'] * transmission_fraction).data)
    denominator.coords['position'] = data.meta['position']
    denominator.coords['gravity'] = data.meta['gravity']
    denominator.coords['sample_position'] = data.meta['sample_position']
    denominator.coords['source_position'] = data.meta['source_position']

    # 3. Convert from wavelength to Q ==================================================

    data = data.transform_coords("Q", graph=graph)
    q_boundaries = sc.concat([q_bins.min(), q_bins.max()], dim='Q')
    q_binned = sc.bin(data, edges=[q_boundaries])
    q_summed = q_binned.bins.concat('spectrum')

    den_q = denominator.transform_coords("Q", graph=graph)
    den_hist = sc.histogram(den_q, bins=q_bins)
    den_q_summed = den_hist.sum('spectrum')

    # 4. Normalize =====================================================================

    return q_summed.bins / sc.lookup(func=den_q_summed, dim='Q')
