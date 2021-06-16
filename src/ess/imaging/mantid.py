import scippneutron as scn
import scipp as sc
import numpy as np


def load_component_info_to_2d(geometry_file, advanced_geometry=False):
    """Load geometry information from a mantid Instrument Definition File
        or a NeXuS file containing instrument geometry. and reshape into 2D
        physical dimensions.

        This function requires mantid-framework to be installed

        :param geometry_file: IDF or NeXus file
        :return dictionary of component names to positions,
            rotations and shapes
        :raises ImportError if mantid cannot be imported
        """
    from mantid.simpleapi import LoadEmptyInstrument
    ws = LoadEmptyInstrument(Filename=geometry_file, StoreInADS=False)
    source_pos, sample_pos = scn.mantid.make_component_info(ws)
    geometry = {}
    geometry["source_position"] = source_pos
    geometry["sample_position"] = sample_pos
    print(source_pos)
    print(sample_pos)
    pos, rot, shp = scn.mantid.get_detector_properties(
        ws,
        source_pos,
        sample_pos,
        spectrum_dim='spectrum',
        advanced_geometry=advanced_geometry)
    x = int(np.sqrt(pos.shape[0]))
    pos2d = sc.fold(pos, dim='spectrum', dims=['y', 'x'], shape=[x, x])
    geometry["position"] = pos2d
    if rot is not None:
        rot2d = sc.fold(rot, dim='spectrum', dims=['y', 'x'], shape=[x, x])
        geometry["rotation"] = rot2d
    if shp is not None:
        shp2d = sc.fold(shp, dim='spectrum', dims=['y', 'x'], shape=[x, x])
        geometry["shape"] = shp2d
    geometry["x"] = pos2d.fields.x['y', 0]
    geometry["y"] = pos2d.fields.y['x', 0]
    return geometry
