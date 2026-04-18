"""Physical and astronomical constants (SI).

CODATA values via `scipy.constants`. Solar nominal radii/masses follow IAU 2015
Resolution B3 (nominal solar GM and radius); solar mass is derived from nominal
GM and CODATA `G` so that `G * M_sun` equals the nominal GM exactly.
"""

from __future__ import annotations

import scipy.constants as sc

# --- CODATA / scipy (values track installed SciPy/CODATA revision) ---
c: float = sc.c
G: float = sc.G
h: float = sc.h
k: float = sc.k
au: float = sc.au
pc: float = sc.parsec

# IAU 2015: nominal solar GM (exact) [m^3 s^-2]
_GM_SUN_NOMINAL_SI: float = 1.32712440042e20

# IAU 2015: nominal solar radius (exact) [m]
_R_SUN_NOMINAL_SI: float = 6.957e8

R_sun: float = _R_SUN_NOMINAL_SI
M_sun: float = _GM_SUN_NOMINAL_SI / G


def gm_sun_si() -> float:
    """Standard gravitational parameter of the Sun, GM_sun [m^3 s^-2]."""
    return G * M_sun
