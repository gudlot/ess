# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
This is the class for data reduction from the Amor instrument, which is a subclass of the broader `ReflData` class.
Features of this class included correcting for the time-of-flight measurement at Amor.
"""
# import copy
import numpy as np
import scipp as sc
# import scippneutron as scn
# from ess.reflectometry import HDM, orso, write
# from ess.reflectometry.data import ReflData
from ... import reflectometry as refl


def _tof_correction(data, tau, chopper_phase):
    """
    A correction for the presense of the chopper with respect to the "true" ToF.
    Also fold the two pulses.
    TODO: generalise mechanism to fold any number of pulses.
    """
    tof_offset = tau * chopper_phase / (180.0 * sc.units.deg)
    # Make 2 bins, one for each pulse
    edges = sc.array(dims=['tof'],
                     values=[
                         -tof_offset.value, (tau - tof_offset).value,
                         (2 * tau - tof_offset).value
                     ],
                     unit=tau.unit)
    data = sc.bin(data, edges=[edges])
    # Make one offset for each bin
    offset = sc.concatenate(tof_offset, tof_offset - tau, 'tof')
    # Apply the offset on both bins
    data.bins.coords['tof'] += offset
    # Rebin to exclude second (empty) pulse range
    return sc.bin(data, edges=[sc.concatenate(0. * sc.units.us, tau, 'tof')])


def load(filename,
         reduction_creator=None,
         data_owner=None,
         experiment_id=None,
         experiment_date=None,
         sample_description=None,
         reduction_file=None,
         data_file=None,
         reduction_creator_affiliation=None,
         sample_angle_offset=0 * sc.units.deg,
         gravity=True,
         beam_size=0.001 * sc.units.m,
         sample_size=0.01 * sc.units.m,
         detector_spatial_resolution=0.0025 * sc.units.m,
         chopper_sample_distance=15.0 * sc.units.m,
         chopper_speed=20 / 3 * 1e-6 / sc.units.us,
         chopper_detector_distance=19.0 * sc.units.m,
         chopper_chopper_distance=0.49 * sc.units.m,
         chopper_phase=-8.0 * sc.units.deg,
         wavelength_bins=sc.linspace(dim='wavelength',
                                     start=2.4,
                                     stop=15.0,
                                     num=201,
                                     unit=sc.units.angstrom)):
    """
    Loader for a single Amor data file.

    Args:
        data (:py:class:`scipp._scipp.core.DataArray` or :py:attr:`str`): The data to be reduced or the path to the file to be reduced.
        reduction_creator (:py:attr:`str`, optional): The name of the creator of the reduction. Optional, default :code:`None`.
        data_owner (:py:attr:`str`, optional): The name of the owner of the data. Optional, default :code:`None`.
        experiment_id (:py:attr:`str`, optional): The experimental identifier. Optional, default :code:`None`.
        experiment_date (:py:attr:`str`, optional): The date or date range for the experiment. Optional, default :code:`None`.
        sample_description (:py:attr:`str`, optional): A short description of the sample. Optional, default :code:`None`.
        reduction_file (:py:attr:`str`, optional): The name of the file used for reduction (:code:`.py` script or :code:`.ipynb` notebook). Optional, default :code:`None`.
        data_file (:py:attr:`str`, optional): If a :py:class:`scipp._scipp.core.DataArray` is given as the :py:attr:`data`, a :py:attr:`data_file` should be defined for output in the file. Optional, default :code:`None`.
        reduction_creator_affiliation (:py:attr:`str`, optional): The affiliation of the reduction owner. Optional, defaults to :code:`None`.
        sample_angle_offset (:py:class:`scipp.Variable`, optional): Correction for omega or possibly misalignment of sample. Optional, default :code:`0 degrees of arc`.
        gravity (:py:attr:`bool`, optional): Should gravity be accounted for. Optional, default `True`.
        beam_size (:py:class:`scipp._scipp.core.Variable`, optional): Size of the beam perpendicular to the scattering surface. Optional, default :code:`0.001 m`.
        sample_size (:py:class:`scipp._scipp.core.Variable`, optional): Size of the sample in direction of the beam. Optional, default :code:`0.01 m`.
        detector_spatial_resolution (:py:class:`scipp._scipp.core.Variable`, optional): Spatial resolution of the detector. Optional, default :code:`2.5 mm`
        chopper_sample_distance (:py:class:`scipp._scipp.core.Variable`, optional): Distance from chopper to sample. Optional, default :code:`15. m`
        chopper_speed (:py:class:`scipp._scipp.core.Variable`, optional): Rotational velocity of the chopper. Optional, default :code:`6.6666... e-6 µs^{-1}`.
        chopper_detector_distance (:py:class:`scipp._scipp.core.Variable`, optional): Distance from chopper to detector. Optional, default :code:`19 m`.
        chopper_chopper_distance (:py:class:`scipp._scipp.core.Variable`, optional): The distance between the wavelength defining choppers. Optional, default :code:`0.49 m`
        chopper_phase (:py:class:`scipp._scipp.core.Variable`, optional): Phase offset between chopper pulse and ToF zero. Optional, default :code:`-8. degrees of arc`.
        wavelength_cut (:py:class:`scipp._scipp.core.Variable`, optional): Minimum cutoff for wavelength. Optional, default :code:`2.4 Å`.

    Attributes:
        tau (:py:class:`scipp._scipp.core.Variable`): Half of the inverse of the chopper speed.
    """
    data = refl.io.load(filename=filename,
                        sample_angle_offset=sample_angle_offset,
                        gravity=gravity,
                        beam_size=beam_size,
                        sample_size=sample_size,
                        detector_spatial_resolution=detector_spatial_resolution)

    # Convert tof nanoseconds to microseconds for convenience
    # TODO: is it safe to assume that the dtype of the binned wrapper coordinate is
    # the same as the dtype of the underlying event coordinate?
    if data.coords['tof'].dtype != sc.dtype.float64:
        data.bins.coords['tof'] = data.bins.coords['tof'].astype('float64')
        data.coords['tof'] = data.coords['tof'].astype('float64')
    data.bins.coords['tof'] = sc.to_unit(data.bins.coords['tof'], 'us')
    data.coords['tof'] = sc.to_unit(data.coords['tof'], 'us')

    # These are Amor specific parameters
    tau = 1 / (2 * chopper_speed)

    # # TODO: do we still need this as attributes?
    # chopper_detector_distance = chopper_detector_distance
    # chopper_chopper_distance = chopper_chopper_distance
    # chopper_phase = chopper_phase
    # wavelength_cut = wavelength_cut

    # The source position is not the true source position due to the
    # use of choppers to define the pulse.
    data.coords["source_position"] = sc.geometry.position(0.0 * sc.units.m,
                                                          0.0 * sc.units.m,
                                                          -chopper_sample_distance)
    # tof_correction()
    data = _tof_correction(data, tau, chopper_phase)
    # The wavelength contribution to the resolution function, defined
    # by the distance between the two choppers.
    # Division by 2np.sqrt(2*np.log(2)) converts from FWHM to std.
    data.coords["sigma_lambda_by_lambda"] = chopper_chopper_distance / (
        data.coords["position"].fields.z - data.coords["source_position"].fields.z)
    data.coords["sigma_lambda_by_lambda"] /= 2 * np.sqrt(2 * np.log(2))

    data = refl.utils.to_wavelength(data, wavelength_bins=wavelength_bins)

    data.coords["velocity"] = refl.utils.compute_velocity(data.coords["wavelength"],
                                                          "m/s")

    attrs = refl.utils.compute_theta(data)
    for name, attr in attrs.items():
        data.coords[name] = attr

    # data /= refl.corrections.illumination_correction(beam_size, sample_size,
    #                                                  data.bins.coords["theta"])
    # illumination()
    data.coords["qz"] = refl.utils.compute_qz(theta=data.coords["theta"],
                                              wavelength=data.coords["wavelength"])
    # find_qz()
    # _setup_orso(reduction_creator, reduction_creator_affiliation, sample_description,
    #             data_owner, experiment_id, experiment_date, reduction_file)
    return data
