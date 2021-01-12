"""
File containing physical constants in SI Units (for conversion between CGS and SI) and useful
equations for dealing with gaussian beams
"""
import numpy as np
from numpy import pi

# Fundamental Constants
# Format: constant = number #desciption; SI unit symbol
hb = 1.0545718e-34  # hbar, reduced plank constant; J*s
h = 2*pi*hb  # planck constant; J*s
e = 1.6021766208e-19  # electron charge; C
me = 9.10938356e-31  # electron mass; kg
alpha = 1./137.035999  # fs constant
c = 2.99792458e8  # speed of light; m/s
eps = 8.854187e-12  # Vacuum Permittivity; F/m
ke = 1./(4*pi*eps)  # Coulomb constant; m/F
ao = 0.52917721067e-10  # bohr radius; m
mu = 1/(c**2*eps)  # Vacuum Permeability; N/A^2
mub = 927.4009994e-26  # Bohr Magneton; J/T
Eh = 4.359744650e-18  # Hartree; J
kb = 1.38064852e-23  # Boltzmann Constant; J/K


# useful functions for electromagnetic waves
def k_mag(omega: float) -> float:
    """
    Returns the magnitude of the k-vector an electromagnetic wave travelling in vacuum with an
        angular frequency of omega
    Args:
        omega : angular frequency of the travelling wave. Use Hz if in SI

    Returns:
        Magnitude of the k-vector an electromagnetic wave travelling in vacuum with an
        angular frequency of omega. In SI (1/m)
    """
    return omega / c


def freq(wavelength: float) -> float:
    return c / wavelength


# nearly ubiquitously useful functions for gaussian beams
def intensity(power: float, width: float) -> float:
    """
    Intensity of a gaussian beam
    Args:
        power : power of the beam
        width : width of the beam

    Returns:
        intensity of the beam in units of P/W**2 where P and W are the units of power and width
        passed in. In other words, if power is given in Watts, and width in centimeters,
        units of intensity returned are Watts/centimeter**2
    """

    return 2*power/(pi*width**2)


def electric_field_strength(power: float, width: float):
    """
    Amplitude of electric field oscillations at the peak of gaussian beam
    Args:
        power : power of the gaussian beam in question (Watts)
        width : width of the gaussian beam in question (Meters)

    Returns:
        electric field strength at the center of a gaussian beam with given power and width in SI
            units
    """
    return np.sqrt(2 * intensity(power, width) / (c * eps))
