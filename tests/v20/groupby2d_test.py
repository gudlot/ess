import numpy as np
import scipp as sc
from ess.v20.imaging.operations import groupby2D


def test_groupby2d():
    data = sc.array(dims=['wavelength', 'x', 'y'],
                    values=np.arange(100.0).reshape(1, 10, 10))
    wav = sc.scalar(value=1.0)
    x = sc.array(dims=['x'], values=np.arange(10))
    y = sc.array(dims=['y'], values=np.arange(10))
    source_position = sc.scalar(value=[0, 0, -10],
                                dtype=sc.dtype.vector_3_float64)
    ds = sc.Dataset(data={'a': data},
                    coords={
                        'x': x,
                        'y': y,
                        'wavelength': wav,
                        'source_position': source_position
                    })
    grouped = groupby2D(ds, 5)
    assert grouped['a'].shape == [1, 5, 5]
    grouped = groupby2D(ds, 1)
    assert grouped['a'].shape == [1, 1, 1]
