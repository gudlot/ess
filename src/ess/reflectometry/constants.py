# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
import scipp as sc
import scipp.constants as constants

# h = (h * 1e20 / 1e6 *
#      (sc.units.kg * sc.units.angstrom * sc.units.angstrom / sc.units.us))
HDM = constants.h / constants.neutron_mass
G_ACC = sc.geometry.position(0. * constants.G.unit, constants.G, 0. * constants.G.unit)
