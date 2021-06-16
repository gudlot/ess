from ess.imaging import mantid
import tempfile
import os
import pytest


@pytest.fixture(scope="module")
def geom_file():
    import mantid.simpleapi as sapi
    # 100 output positions (10 by 10)
    ws = sapi.CreateSampleWorkspace(NumBanks=1,
                                    BankPixelWidth=10,
                                    StoreInADS=False)
    file_name = "example_geometry.nxs"
    geom_path = os.path.join(tempfile.gettempdir(), file_name)
    sapi.SaveNexusGeometry(ws, geom_path)
    assert os.path.isfile(geom_path)  # sanity check
    yield geom_path
    try:
        os.remove(geom_path)
    except:
        pass


def test_load_component_info_to_2d_geometry_bad_sizes(geom_file):
    bad_sizes = {'x': 10, 'y': 5}  # gives volume of 50 not 100
    with pytest.raises(ValueError):
        mantid.load_component_info_to_2d(geom_file, sizes=bad_sizes)


def test_load_component_info_to_2d_geometry(geom_file):
    geometry = mantid.load_component_info_to_2d(geom_file,
                                                sizes={
                                                    'x': 10,
                                                    'y': 10
                                                })
    assert geometry["position"].sizes == {'x': 10, 'y': 10}


def test_load_component_info_to_2d_geometry_irregular(geom_file):
    geometry = mantid.load_component_info_to_2d(geom_file,
                                                sizes={
                                                    'y': 2,
                                                    'x': 50
                                                })
    assert geometry["position"].sizes == {'x': 50, 'y': 2}


def test_load_component_info_to_2d_geometry_with_specified_keys(geom_file):
    geometry = mantid.load_component_info_to_2d(geom_file,
                                                sizes={
                                                    'u': 10,
                                                    'v': 10
                                                })
    assert geometry["position"].sizes == {'u': 10, 'v': 10}


def test_load_component_info_to_2d_geometry_advanced_geom(geom_file):
    geometry = mantid.load_component_info_to_2d(geom_file,
                                                sizes={
                                                    'x': 10,
                                                    'y': 10
                                                },
                                                advanced_geometry=True)
    assert geometry["position"].sizes == {'x': 10, 'y': 10}
    assert geometry["rotation"].sizes == {'x': 10, 'y': 10}
    assert geometry["shape"].sizes == {'x': 10, 'y': 10}
