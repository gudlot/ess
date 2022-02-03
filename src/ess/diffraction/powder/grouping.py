# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2022 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scippneutron as scn


def focus_by_two_theta(data, *, bins, replace_dim='spectrum'):
    data = data.copy(deep=False)
    if 'two_theta' not in data.meta and 'two_theta' not in data.bins.meta:
        data.coords['two_theta'] = scn.two_theta(data)
    return sc.groupby(data, 'two_theta', bins=bins).bins.concat(replace_dim)
