"""Sanity checks on physical constants (SciPy-backed)."""

from oblateness import constants as C


def test_speed_of_light_order():
    assert 2.99e8 < C.c < 3.01e8


def test_gravitational_constant_order():
    assert 6.6e-11 < C.G < 6.8e-11


def test_au_meter_scale():
    assert 1.4e11 < C.au < 1.6e11


def test_gm_sun_matches_nominal():
    # IAU 2015 nominal GM; must match derived G * M_sun in constants.py
    nominal = 1.32712440042e20
    assert C.gm_sun_si() == nominal
