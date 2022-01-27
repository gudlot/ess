from ess.imaging import mantid
import tempfile
import os
import pytest


def mantid_is_available():
    try:
        import mantid  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


# with_mantid_only = pytest.mark.skipif(not mantid_is_available(),
#                                       reason='Mantid framework is unavailable')

# @pytest.fixture(scope="module")
# def geom_file():
#     import mantid.simpleapi as sapi
#     # 100 output positions (10 by 10)
#     ws = sapi.CreateSampleWorkspace(NumBanks=1,
#                                     BankPixelWidth=10,
#                                     PixelSpacing=0.01,
#                                     StoreInADS=False)
#     file_name = "example_geometry.nxs"
#     geom_path = os.path.join(tempfile.gettempdir(), file_name)
#     sapi.SaveNexusGeometry(ws, geom_path)
#     assert os.path.isfile(geom_path)  # sanity check
#     yield geom_path
#     try:
#         os.remove(geom_path)
#     except Exception:
#         pass

# @with_mantid_only
# def test_load_component_info_to_2d_geometry_bad_sizes(geom_file):
#     bad_sizes = {'x': 10, 'y': 5}  # gives volume of 50 not 100
#     with pytest.raises(ValueError):
#         mantid.load_component_info_to_2d(geom_file, sizes=bad_sizes)
