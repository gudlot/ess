import numpy as np
import scipp as sc
from ess.v20.imaging.operations import groupby2D


def test_groupby2d_simple_case():
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
    grouped = groupby2D(ds, 5, 5)
    assert grouped['a'].shape == [1, 5, 5]
    grouped = groupby2D(ds, 1, 1)
    assert grouped['a'].shape == [1, 1, 1]


def test_any_naming():
    data = sc.array(dims=['u', 'v', 'w'],
                    values=np.arange(200.0).reshape(2, 10, 10))
    u = sc.array(dims=['u'], values=np.arange(2))
    v = sc.array(dims=['v'], values=np.arange(10))
    w = sc.array(dims=['w'], values=np.arange(10))
    ds = sc.Dataset(data={'a': data}, coords={
        'v': v,
        'w': w,
        'u': u,
    })
    grouped = groupby2D(ds,
                        nx_target=5,
                        ny_target=5,
                        x='v',
                        y='w',
                        z='u',
                        preserve=[])
    assert grouped['a'].shape == [2, 5, 5]


def test_groupby2d_different_output_size():
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
    grouped = groupby2D(ds, nx_target=2, ny_target=5)
    assert grouped['a'].sizes['x'] == 2
    assert grouped['a'].sizes['y'] == 5
    grouped = groupby2D(ds, nx_target=10, ny_target=1)
    assert grouped['a'].sizes['x'] == 10
    assert grouped['a'].sizes['y'] == 1
