import scipp as sc
from scipy.constants import neutron_mass, h, g

h = h * 1e20 / 1e6 * (sc.units.kg * sc.units.angstrom * sc.units.angstrom / sc.units.us)
HDM = (h / (neutron_mass * sc.units.kg))
G_ACC = -g * (sc.units.m / (sc.units.s * sc.units.s))
