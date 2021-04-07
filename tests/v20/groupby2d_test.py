import numpy as np
import scipp as sc
from ess.v20.imaging.operations import groupby2D


def test_groupby2d_simple_case_neutron_specific():
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


def _make_simple_dataset(u, v, w):
    size = np.prod((u, v, w))
    data = sc.array(dims=['u', 'v', 'w'],
                    values=np.arange(size).reshape((u, v, w)))
    u = sc.array(dims=['u'], values=np.arange(u))
    v = sc.array(dims=['v'], values=np.arange(v))
    w = sc.array(dims=['w'], values=np.arange(w))
    return sc.Dataset(data={'a': data}, coords={
        'v': v,
        'w': w,
        'u': u,
    })


def test_simple_case_any_naming():
    ds = _make_simple_dataset(u=2, v=10, w=10)
    grouped = groupby2D(ds,
                        nx_target=5,
                        ny_target=5,
                        x='v',
                        y='w',
                        z='u',
                        preserve=[])
    assert grouped['a'].shape == [2, 5, 5]


def test_groupby2d_different_output_size():
    ds = _make_simple_dataset(u=2, v=10, w=10)
    grouped = groupby2D(ds,
                        nx_target=2,
                        ny_target=5,
                        x='v',
                        y='w',
                        z='u',
                        preserve=[])
    assert grouped['a'].sizes['v'] == 2
    assert grouped['a'].sizes['w'] == 5
    assert grouped['a'].sizes['u'] == 2
