def normalize_uv(
    *, sample: sc.DataArray, reference: sc.DataArray, dark: sc.DataArray
) -> sc.DataArray: 
    """Calculates the absorbance of the UV signal.

    Parameters
    ----------
    sample: sc.DataArray
        DataArray containing sample UV signal, one spectrum or multiple spectra.
    reference: sc.DataArray
        DataArray containing reference UV signal, one spectrum expected.
    dark: sc.DataArray
        DataArray containing dark UV signal, one spectrum expected.

    Returns
    ----------
    normalized: sc.DataArray
        DataArray that contains the normalized UV signal, one spectrum or mulitple spectra.

    """

    normalized = sc.log10(
        (reference - dark) / (sample - dark)
    )  # results in DataArrays with multiple spectra

    return normalized

def load_and_normalize_uv(name):
    """Loads the UV data from the corresponding entry in the LoKI.nxs filename and 
    calculates the absorbance of each UV spectrum.
    For an averaged spectrum based on all UV spectra in the file, use process_uv.

    Parameters
    ----------
    name: str
        Filename, e.g. 066017.nxs

    Returns:
    ----------
    normalized: sc.DataArray
        DataArray that contains the normalized UV signal, one spectrum or mulitple spectra.

    """
    uv_dict = load_nurfloki_file(name, 'uv')
    normalized = normalize_uv(**uv_dict)  # results in DataArrays with multiple spectra

    return normalized

def process_uv(name):
    """Processses all UV spectra in a single LoKI.nxs and averages them to one corrected
       UV spectrum.

    Parameters
    ----------
    name: str
        Filename for a LoKI.nxs file containting UV entry.

     Returns
    ----------
    normalized: 
        One averaged UV spectrum. Averaged over all UV spectra contained in the file
        under UV entry data.

    """

    uv_dict = load_nurfloki_file(name, 'uv')
    normalized = normalize_uv(**uv_dict) 

    # returns averaged uv spectrum
    return normalized.mean("spectrum")

def gather_uv_set(flist_num):
    """Creates a sc.DataSet for set of given filenames for an experiment composed of 
    multiple, separated UV measurements over time.
    Parameters
    ----------
    flist_num: list of int
        List of filenames as numbers (ILL style) containing UV data

     Returns
    ----------
    uv_spectra_set: sc.DataSet
        DataSet of multiple UV DataArrays, where the UV signal for each experiment was averaged

    """

    uv_spectra_set = sc.Dataset({name: process_uv(name) for name in flist_num})
    return uv_spectra_set


def uv_peak_int(uv_da: sc.DataArray, wavelength=None, wl_unit=None, tol=None):
    """Extract uv peak intensity for a given wavelength [wl_unit] and a given interval [wl_unit].
    First version: interval around wavelength of width 2*tol, values are averaged and then we need interpolation to get value at requested wavelength??? not sure yet how to realise this
    If no wavelength is given, 280 is chosen as value
    If no wl_unit is given, the unit of the wavelength coordinate of the sc.DataArray is chosen, e.g. [nm]. Other option to generate a unit: wl_unit=sc.Unit('nm')

    Parameters
    ----------
    uv_da: sc.DataArray
        DataArray containing uv spectra
    wavelength: float
        Wavelength
    wl_unit: sc.Unit
        Unit of the wavelength
    tol: float
        Tolerance, 2*tol defines the interval around the given wavelength

    Returns
    ----------
    uv_peak_int: dict
        Dictionary that contains the peak intensity for the requested wavelength, the peak intensity averaged over the requested interval, the requested wavelength with its unit, and the tolerance

    """
    assert (
        "wavelength" in uv_da.dims
    ), "sc.DataArray is missing the wavelength dimension"  # assert that 'wavelength' is a dimension in the uv_da sc.DataArray

    # obtain unit of wavelength:
    if wl_unit is None:
        wl_unit = uv_da.coords["wavelength"].unit
    else:
        if not isinstance(wl_unit, sc.Unit):
            raise TypeError
        assert (
            wl_unit == uv_da.coords["wavelength"].unit
        )  # we check that the given unit corresponds to the unit for the wavelength

    # set default value for wavelength:
    if wavelength is None:
        wavelength = 280

    # set default value for tolerance:
    if tol is None:
        tol = 0.5

    # filter spectrum values for the specified interval, filtered along the wavelength dimension
    uv_da_filt = uv_da[
        "wavelength",
        (wavelength - tol) * wl_unit : (wavelength + tol) * wl_unit,
    ]

    # try out interpolation
    from scipp.interpolate import interp1d

    uv_interp = interp1d(uv_da, "wavelength")
    # get new x values, in particular I want the value at one wavelength
    x2 = sc.linspace(
        dim="wavelength", start=wavelength, stop=wavelength, num=1, unit=wl_unit
    )
    # alternative
    # x2 = sc.array(dims=['wavelength'], values=[wavelength], unit=wl_unit)
    # this gives us the intensity value for one specific wavelength
    uv_int_one_wl = uv_interp(x2)

    # now we want to have as well the wavelength for the interval
    uv_int_mean_interval = uv_da_filt.mean(dim="wavelength")

    # prepare a dict for output
    uv_peak_int = {
        "one_wavelength": uv_int_one_wl,
        "wl_interval": uv_int_mean_interval,
        "wavelength": wavelength,
        "unit": wl_unit,
        "tol": tol,
    }

    return uv_peak_int

def turbidity(wl, b, m):
    """Function describing turbidity tau. tau = b* lambda **(-m)
        Fitting parameters: b, m. b corresponds to the baseline found for higher wavelengths (flat line in UV spectrum), m corresponds to the slope.
        lambda: wavelength

    Parameters
    ----------
    b: np.ndarray
        Offset, baseline

    m: np.ndarray
        Slope

    wl: np.ndarray
        UV wavelengths

    Returns
    ----------
    y: np.ndarray
        Turbidity

    """

    y = b * wl ** (-m)
    return y


def residual(p, x, y):
    """Calculates the residuals between fitted turbidity and measured UV data

    Parameters
    ----------
    p: list
        Fit parameters for turbidity

    x: np.ndarray
        x values, here: UV wavelength

    y: np.ndarray
        y values, here: UV intensity

    Returns
    ----------
    y - turbidity(x, *p): np.ndarray
        Difference between measured UV intensity values and fitted turbidity

    """

    return y - turbidity(x, *p)




def uv_turbidity_fit(
    uv_da: sc.DataArray,
    wl_unit=None,
    fit_llim=None,
    fit_ulim=None,
    b_llim=None,
    b_ulim=None,
    m=None,
    plot_corrections=True,
):
    """Fit turbidity to the experimental data. Turbidity: tau=b * wavelength^(-m) Parameters of interest: b, m.
        b is the baseline and m is the slope. b can be obtained by averaging over the flat range of the UV spectrum
        in the higher wavelength range.
        m: make an educated guess. Advice: Limit fitting range to wavelengths after spectroscopy peaks.

    Parameters
    ----------
    uv_da: sc.DataArray
        UV sc.DataArray containing one or more normalized UV spectra
    wl_unit: sc.Unit
        Wavelength unit
    fit_llim: int
        Lower wavelength limit of fit range for turbidity
    fit_ulim: int
        Upper wavelength limit of fit range for turbidity
    b_llim: int
        Lower wavelength limit of fit range for b
    b_ulim: int
        Upper wavelength limit of fit range for b
    m: int
        Educated guess start value of slope parameter in turbidity
    plot_corrections: bool
        If true, plot single contribitions for turbidity corrections

    Returns:
    ----------
    uv_da_turbcorr: sc.DataArray
        uv_da dataarray where each spectrum was corrected for a fitted turbidity, export for all wavelength values

    """
    # obtain unit of wavelength:
    if wl_unit is None:
        wl_unit = uv_da.coords["wavelength"].unit
    else:
        if not isinstance(wl_unit, sc.Unit):
            raise TypeError
        assert (
            wl_unit == uv_da.coords["wavelength"].unit
        )  # we check that the given unit corresponds to the unit for the wavelength

    if fit_llim is None:
        fit_llim = 400
    if fit_ulim is None:
        fit_ulim = 800

    if fit_llim is not None and fit_ulim is not None:
        assert fit_llim < fit_ulim, "fit_llim < fit_ulim"
    if b_llim is not None and b_ulim is not None:
        assert b_llim < b_ulim, "b_llim<b_ulim"

    if m is None:
        m = 0.01

    # select the UV wavelength range for fitting the turbidity
    uv_da_filt = uv_da["wavelength", fit_llim * wl_unit : fit_ulim * wl_unit]

    # How many spectra are in the file?
    num_spec = uv_da_filt.sizes["spectrum"]

    # offset, choose wavelength range for b0 extraction and average over the slected wavelength range
    b0 = (
        uv_da["wavelength", b_llim * wl_unit : b_ulim * wl_unit]
        .mean(dim="wavelength")
        .values
    )
    # create np.ndarray of same shape as b, but with values of m
    m0 = np.full(b0.shape, m)

    # create dummy array
    uv_da_turb_corr_dat = np.zeros(uv_da.data.shape)

    res_popt = np.empty([num_spec, 2])
    res_pcov = np.empty([num_spec, 1])
    # Perform the fitting
    for i in range(num_spec):
        # start parameters
        p0 = [b0[i], m0[i]]
        popt, pcov = leastsq(
            residual,
            p0,
            args=(
                uv_da_filt.coords["wavelength"].values,
                uv_da_filt["spectrum", i].values,
            ),
        )

        # calculation of each spectrum corrected for fitted turbidity
        uv_da_turb_corr_dat[i, :] = uv_da["spectrum", i].values - turbidity(
            uv_da["spectrum", i].coords["wavelength"].values, popt[0], popt[1]
        )
        # don't lose fit parameters
        res_popt[i, :] = popt
        res_pcov[i, 0] = pcov

    # Prepare for the new data uv_da corrected for a fitted turbidity
    uv_da_turbcorr = uv_da.copy()
    # Replace data in the dataarray
    # Is this a good way to replace the data in the sc.DataArray? It works and I don't have all methods available like in xarray, but is it slow?
    uv_da_turbcorr.data.values = uv_da_turb_corr_dat

    # Collect the results of the fitting and store them with the dataarray
    uv_da_turbcorr.attrs["fit-slope_m"] = sc.array(
        dims=["spectrum"], values=res_popt[:, 1]
    )
    uv_da_turbcorr.attrs["fit-offset_b"] = sc.array(
        dims=["spectrum"], values=res_popt[:, 0]
    )
    uv_da_turbcorr.attrs["fit-pcov"] = sc.array(
        dims=["spectrum"], values=res_pcov[:, 0]
    )

    # display(uv_da_turbcorr)

    # Switch on plots for verification
    if plot_corrections:

        # Plotting results as sc.plot
        fig2, ax2 = plt.subplots(ncols=2, figsize=(12, 5))
        out0 = sc.plot(
            sc.collapse(uv_da, keep="wavelength"),
            grid=True,
            title="before correction",
            ax=ax2[0],
        )
        out1 = sc.plot(
            sc.collapse(uv_da_turbcorr, keep="wavelength"),
            grid=True,
            title="after correction",
            ax=ax2[1],
        )

        fig, ax = plt.subplots(ncols=num_spec + 1, figsize=(18, 5))
        out3 = sc.plot(
            sc.collapse(uv_da_filt, keep="wavelength"),
            grid=True,
            title="Selection for turbidity",
            ax=ax[-1],
        )

        for i in range(num_spec):
            # collect the fitting parameters for each spectrum to avoid new fitting
            popt = [
                uv_da_turbcorr.attrs["fit-offset_b"]["spectrum", i].values,
                uv_da_turbcorr.attrs["fit-slope_m"]["spectrum", i].values,
            ]

            ax[i].plot(
                uv_da.coords["wavelength"].values,
                uv_da["spectrum", i].values,
                "s",
                label=f"Full UV raw data {i}",
            )
            ax[i].plot(
                uv_da.coords["wavelength"].values,
                uv_da["spectrum", i].values
                - turbidity(uv_da.coords["wavelength"].values, popt[0], popt[1]),
                "x",
                label=f"Whole UV spectrum, fitted turbidity subtracted {i}",
            )
            ax[i].plot(
                uv_da.coords["wavelength"].values,
                turbidity(uv_da.coords["wavelength"].values, popt[0], popt[1]),
                "^",
                label=f"Full turbidity {i}",
            )

            ax[i].plot(
                uv_da_filt.coords["wavelength"].values,
                turbidity(uv_da_filt.coords["wavelength"].values, popt[0], popt[1]),
                ".",
                label=f"Fitted turbidity {i}, b={popt[0]:.3f}, m={popt[1]:.3f}",
            )

            # ax[i].plot(uv_da['wavelength', b_llim*wl_unit: b_ulim*wl_unit]['spectrum',i].coords['wavelength'].values, uv_da['wavelength',
            #   b_llim*wl_unit: b_ulim*wl_unit]['spectrum',i].values,'v', label=f'Selection for b0 {i}')
            # No need to slice in the spectrum dimension for the x values
            ax[i].plot(
                uv_da["wavelength", b_llim * wl_unit : b_ulim * wl_unit]
                .coords["wavelength"]
                .values,
                uv_da["wavelength", b_llim * wl_unit : b_ulim * wl_unit][
                    "spectrum", i
                ].values,
                "v",
                label=f"Selection for b0 {i}",
            )

            ax[i].grid(True)
            ax[i].set_xlabel("Wavelength [nm]")
            ax[i].set_ylabel("Absorbance")
            ax[i].legend()
            ax[i].set_title(f"Spectrum {str(i)}")
            # set limits, np.isfinite filters out inf (and nan) values
            ax[i].set_ylim(
                [
                    -0.5,
                    1.1
                    * uv_da["spectrum", i]
                    .values[np.isfinite(uv_da["spectrum", i].values)]
                    .max(),
                ]
            )

        # display(fig2)

    return uv_da_turbcorr


def uv_multi_turbidity_fit(
    filelist,
    wl_unit=sc.Unit("nm"),
    fit_llim=300,
    fit_ulim=850,
    b_llim=450,
    b_ulim=700,
    m=0.1,
    plot_corrections=False,
):
    """Applies turbidity correction to uv spectra for a set of  LoKI.nxs files."""

    uv_collection = {}
    for name in filelist:
        uv_dict = load_uv(name)
        uv_da = normalize_uv(**uv_dict)
        uv_da_turbcorr = uv_turbidity_fit(
            uv_da,
            wl_unit=wl_unit,
            fit_llim=fit_llim,
            fit_ulim=fit_ulim,
            b_llim=b_llim,
            b_ulim=b_ulim,
            m=m,
            plot_corrections=plot_corrections,
        )
        # append names as attributes
        uv_da_turbcorr.attrs["name"] = sc.array(
            dims=["spectrum"], values=[name] * uv_da_turbcorr.sizes["spectrum"]
        )
        uv_collection[f"{name}"] = uv_da_turbcorr

        # print(name,uv_da_turbcorr.data.shape,uv_da_turbcorr.data.values.ndim  )

    multi_uv_turb_corr_da = sc.concat(
        [uv_collection[f"{name}"] for name in filelist], dim="spectrum"
    )
    display(multi_uv_turb_corr_da)

    fig, ax = plt.subplots(ncols=3, figsize=(21, 7))
    legend_props = {"show": True, "loc": 1}
    num_spectra = multi_uv_turb_corr_da.sizes["spectrum"]

    out = sc.plot(
        sc.collapse(multi_uv_turb_corr_da, keep="wavelength"),
        grid=True,
        ax=ax[0],
        legend=legend_props,
        title=f"All turbidity corrected UV spectra for {num_spectra} spectra",
    )
    ax[0].set_ylim(
        [
            -1,
            1.2
            * multi_uv_turb_corr_da.data.values[
                np.isfinite(multi_uv_turb_corr_da.data.values)
            ].max(),
        ]
    )

    out2 = sc.plot(
        sc.collapse(multi_uv_turb_corr_da.attrs["fit-offset_b"], keep="spectrum"),
        title=f"All fit-offset b for {num_spectra} spectra",
        ax=ax[1],
        grid=True,
    )

    # ax0.set_xticks(np.arange(0,len(filelist),1), labels=[f'{name}' for name in filelist], rotation=90)
    secx = ax[1].secondary_xaxis(-0.2)
    secx.set_xticks(
        np.arange(0, num_spectra, 1),
        labels=[f"{name}" for name in multi_uv_turb_corr_da.attrs["name"].values],
        rotation=90,
    )
    out3 = sc.plot(
        sc.collapse(multi_uv_turb_corr_da.attrs["fit-slope_m"], keep="spectrum"),
        title=f"All fit-slope m for {num_spectra} spectra",
        ax=ax[2],
        grid=True,
    )
    secx2 = ax[2].secondary_xaxis(-0.2)
    secx2.set_xticks(
        np.arange(0, num_spectra, 1),
        labels=[f"{name}" for name in multi_uv_turb_corr_da.attrs["name"].values],
        rotation=90,
    )

    display(fig)

    return multi_uv_turb_corr_da






