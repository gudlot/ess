from ess.v20.wfm.stitch import stitch
import scipp as sc


def test_basic_stitching():
    frames = sc.Dataset()
    shift = -5.0
    frames['left_edges'] = sc.array(dims=['frame'], values=[0.0], unit=sc.units.us)
    frames['right_edges'] = sc.array(dims=['frame'], values=[10.0], unit=sc.units.us)
    frames['shifts'] = sc.array(dims=['frame'], values=[shift], unit=sc.units.us)

    data = sc.DataArray(data=sc.ones(dims=['t'], shape=[100], unit=sc.units.counts),
                        coords={
                            't':
                            sc.linspace(dim='t',
                                        start=0.0,
                                        stop=10.0,
                                        num=101,
                                        unit=sc.units.us)
                        })

    nbins = 10
    stitched = stitch(data=data, dim='t', frames=frames, nbins=10, plot=False)
    # Note dimension change to TOF as well as shift
    assert sc.identical(
        sc.values(stitched),
        sc.DataArray(data=sc.ones(dims=['tof'], shape=[nbins], unit=sc.units.counts) *
                     nbins,
                     coords={
                         'tof':
                         sc.linspace(dim='tof',
                                     start=0.0 - shift,
                                     stop=10.0 - shift,
                                     num=nbins + 1,
                                     unit=sc.units.us)
                     }))
