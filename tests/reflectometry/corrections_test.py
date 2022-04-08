# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import warnings
import scipp as sc
from ess.reflectometry import corrections
from orsopy import fileio


def test_normalize_by_counts():
    """
    Tests the corrections.normalize_by_counts function without
    a orsopy object present.
    """
    N = 50
    values = [1.] * N
    data = sc.Variable(dims=['position'],
                       unit=sc.units.counts,
                       values=values,
                       variances=values)
    array = sc.DataArray(data=data)
    with warnings.catch_warnings(record=True) as w:
        array_normalized = corrections.normalize_by_counts(array)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert 'orsopy' in str(w[-1].message)
    result = sc.DataArray(data=sc.Variable(dims=['position'],
                                           unit=sc.units.dimensionless,
                                           values=[1 / N] * N,
                                           variances=[1 / (N * N) + 1 / (N * N * N)] *
                                           N))
    assert sc.allclose(array_normalized.data, result.data)


def test_normalize_by_counts_orso():
    """
    Tests the corrections.normalize_by_counts function
    with a orsopy object present.
    """
    N = 50
    values = [1.] * N
    data = sc.Variable(dims=['position'],
                       unit=sc.units.counts,
                       values=values,
                       variances=values)
    array = sc.DataArray(data=data, attrs={'orso': sc.scalar(fileio.orso.Orso.empty())})
    array.attrs['orso'].value.reduction.corrections = []
    array_normalized = corrections.normalize_by_counts(array)
    result = sc.DataArray(data=sc.Variable(dims=['position'],
                                           unit=sc.units.dimensionless,
                                           values=[1 / N] * N,
                                           variances=[1 / (N * N) + 1 / (N * N * N)] *
                                           N))
    assert sc.allclose(array_normalized.data, result.data)
    assert 'total counts' in array.attrs['orso'].value.reduction.corrections
