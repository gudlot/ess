# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)

# flake8: noqa

from ess import imaging
import tempfile
import os
import pytest
# import scipp as sc
# import numpy as np


def mantid_is_available():
    try:
        import mantid  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


with_mantid_only = pytest.mark.skipif(not mantid_is_available(),
                                      reason='Mantid framework is unavailable')


@pytest.fixture(scope="module")
def geom_file():
    import mantid.simpleapi as sapi
    # # 100 output positions (10 by 10)
    # ws = sapi.CreateSampleWorkspace(NumBanks=1,
    #                                 BankPixelWidth=10,
    #                                 PixelSpacing=0.01,
    #                                 StoreInADS=False)
    # file_name = "example_geometry.nxs"
    # geom_path = os.path.join(tempfile.gettempdir(), file_name)
    # sapi.SaveNexusGeometry(ws, geom_path)
    # assert os.path.isfile(geom_path)  # sanity check
    # yield geom_path
    # try:
    #     os.remove(geom_path)
    # except Exception:
    #     pass


@with_mantid_only
def test_dummy(geom_file):
    assert True


# @with_mantid_only
# def test_load_component_info_to_2d_geometry_bad_sizes(geom_file):
#     bad_sizes = {'x': 10, 'y': 5}  # gives volume of 50 not 100
#     with pytest.raises(ValueError):
#         imaging.mantid.load_component_info_to_2d(geom_file, sizes=bad_sizes)

# @with_mantid_only
# def test_load_component_info_to_2d_geometry(geom_file):
#     geometry = imaging.mantid.load_component_info_to_2d(geom_file,
#                                                         sizes={
#                                                             'x': 10,
#                                                             'y': 10
#                                                         })
#     assert geometry["position"].sizes == {'x': 10, 'y': 10}
#     assert sc.identical(
#         geometry["x"],
#         sc.DataArray(data=sc.array(
#             dims=["x"], values=np.arange(0.0, 0.1, 0.01), unit=sc.units.m)))
#     assert sc.identical(
#         geometry["y"],
#         sc.DataArray(data=sc.array(
#             dims=["y"], values=np.arange(0.0, 0.1, 0.01), unit=sc.units.m)))

# @with_mantid_only
# def test_load_component_info_to_2d_geometry_irregular(geom_file):
#     geometry = imaging.mantid.load_component_info_to_2d(geom_file,
#                                                         sizes={
#                                                             'y': 2,
#                                                             'x': 50
#                                                         })
#     assert geometry["position"].sizes == {'x': 50, 'y': 2}
#     assert "x" in geometry
#     assert "y" in geometry

# @with_mantid_only
# def test_load_component_info_to_2d_geometry_non_cartesian(geom_file):
#     geometry = imaging.mantid.load_component_info_to_2d(geom_file,
#                                                         sizes={
#                                                             'u': 20,
#                                                             'v': 5
#                                                         })
#     assert geometry["position"].sizes == {'u': 20, 'v': 5}
#     # Cannot extract x, y as fields
#     assert "x" not in geometry
#     assert "y" not in geometry

# @with_mantid_only
# def test_load_component_info_to_2d_geometry_advanced_geom(geom_file):
#     geometry = imaging.mantid.load_component_info_to_2d(geom_file,
#                                                         sizes={
#                                                             'x': 10,
#                                                             'y': 10
#                                                         },
#                                                         advanced_geometry=True)
#     assert geometry["position"].sizes == {'x': 10, 'y': 10}
#     assert geometry["rotation"].sizes == {'x': 10, 'y': 10}
#     assert geometry["shape"].sizes == {'x': 10, 'y': 10}
