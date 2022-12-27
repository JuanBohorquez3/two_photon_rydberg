import numpy as np
from numpy import pi

from rabi_frequencies import *
from basics import *
from atomic_physics import *

"""
A set of functions and constants useful for computing matrix elements in the hybrid excitation 
scheme
"""


# constants and derived functions
q_rad_int = 37.5*ao**2


def d_rme_fs(n):
    """
    Computes the reduced matrix element <5D5/2||er||nrP3/2>
    Args:
        n : principle quantum number for our target rydberg state
    Returns:
        reduced matrix element <5D5/2||er||nrP3/2> in SI units (Cm)
    """
    return 0.0024*ao*e*(n/100.)**(-1.5)


hf_A = -21.24e6  # hyperfine A constant for |5D5/2> level. (Hz)
hf_B = 0.2e6  # hyperfine B constant for |5D5/2> level. (Hz)


DHF = hf_splittings(
    a=hf_A,
    b=hf_B,
    i=7/2,
    j=5/2
)

DHF[0] = 1j

gamma_e = gamma_5D

# Compute rabi frequencies
def q_rme_h(q_rad: float = q_rad_int) -> complex:
    """
    Computes reduced matrix element for the quadrupole transition between the |6S> -> |5D> states,
    given the value of the radial integral between those levels.

    The reduced matrix element is <5D||C_2/sqrt(15)||6S>.
    Args:
        q_rad : numerical value of the radial integral between the ground and excited states.
            Expected to be in SI units (m**2)
    Returns:
        q_rme : the reduced matrix element for the quadrupole transition in the nl basis. In SI
            units (Cm**2)
    """
    # Quadrupole excitation reduced matrix element in orbital angular momentum basis
    return e * clebsch_gordan(0, 2, 2, 0, 0, 0) * q_rad / np.sqrt(15)


def q_rme_fs_h(q_rad: float) -> complex:
    """
    Computes reduced matrix element for the quadrupole transition between the |6S1/2> -> |5D5/2>
    states, given the value of the radial integral between those levels.

    The reduced matrix element is <5D5/2||C_2/sqrt(15)||6S1/2>.
    Args:
        q_rad : numerical value of the radial integral between the ground and excited states.
            Expected to be in SI units (m**2)
    Returns:
        q_rme : the reduced matrix element for the quadrupole transition in the fine structure basis
            In SI units (Cm**2)
    """
    return quadrupole_rme_fs(
        q_rme_h(q_rad),
        0,
        1 / 2,
        2,
        5 / 2,
        1 / 2
    )


def q_rme_hf_h(q_rad: float, fp: moment) -> complex:
    """
    Computes reduced matrix element for the quadrupole transition between the |6S> -> |5D> states,
    given the value of the radial integral between those levels.

    The reduced matrix element is <5D5/2;fp||C_2/sqrt(15)||6S1/2;f=4>.
    Args:
        q_rad : numerical value of the radial integral between the ground and excited states.
            Expected to be in SI units (m**2)
        fp: hyperfine structure angular momentum quantum number for excited state. Should be integer
            or half integer
    Returns:
        q_rme : the reduced matrix element for the quadrupole transition in the hyperfine structure
            basis. In SI units (Cm**2)
    """
    return quadrupole_rme_hf(
        q_rme_fs_h(q_rad),
        1 / 2,
        4,
        5 / 2,
        fp,
        7 / 2
    )


def q_rabi_frequency(
        power: float,
        width: float,
        pol_ar: SphericalVector,
        k_ar: SphericalVector,
        q_rad: float,
        fp: moment,
        mp: moment,
        phi: float = 0.0
) -> complex:
    """
    Computes the Rabi frequency between the Cesium states |6S1/2;f=4,m=0> and |5D5/2;fp,mp> through
    the quadrupole transition at 684nm given parameters about the field, and the value of the
    radial integral between the two electron wavefunctions.

    Args:
        power: power of the laser field driving the rabi oscillation. (W)
        width: witdth of the laser field mode driving the rabi oscillation. (m)
        pol_ar: SphericalVector object describing the polarization of the field driving the rabi
            oscillation.
        k_ar: SphericalVector object describing the direction of the k-vector of the field driving
            the rabi oscillation.
        q_rad: value of the radial integral between the ground and excited state. (m**2)
        fp: hyperfine angular momentum quantum number of the excited state. Should be integer or
            half-integer
        mp: aziumthal angular momentum quantum number of the excited state. Should be integer or
            half-integer
        phi: phase of the electric field in radians. Default is 0.
    Returns:
        Rabi frequency coupling the ground state to the given excited state. (Hz)
    """
    e_field = electric_field_strength(power, width) * np.exp(1j * phi)
    rme = q_rme_hf_h(q_rad, fp)
    return quadrupole_rabi_frequency(
        e_field,
        freq(684e-9),
        pol_ar,
        k_ar,
        rme,
        4,
        0,
        fp,
        mp
    )


def d_rabi_frequency(
        power: float,
        width: float,
        pol_ar: SphericalVector,
        nr: int,
        fp: moment,
        mp: moment,
        mr: moment,
        phi: float = 0.0
) -> complex:
    """
    Computes the rabi frequency between the Cesium states |5D5/2;fp,mp> and |nrP3/2mr;7/2> where the
    lower state (primed state) is defined in the hyperfine basis, and the uppper state (rydberg
    state) is defined in the fine structure basis

    Args:
        power: power of the laser field driving the rabi oscillation. (W)
        width: witdth of the laser field mode driving the rabi oscillation. (m)
        pol_ar: SphericalVector object describing the polarization of the field driving the rabi
            oscillation.
        nr: principle quantum number of our target rydberg state
        fp: hyperfine angular momentum quantum number of our primed state
        mp: azimuthal angular momentum quantum number of our primed state
        mr: azimuthal angular momentum quantum number of our rydberg state
        phi: phase of the electric field in radians. Default is 0.
    Returns:
        Rabi frequency coupling the given excited state to the given rydberg state. (Hz)
    """
    e_field = electric_field_strength(power, width) * np.exp(1j * phi)
    rme = d_rme_fs(nr)
    return dipole_hf_to_fs(
        e_field,
        pol_ar,
        rme,
        5 / 2,
        fp,
        mp,
        3 / 2,
        mr,
        7 / 2
    )


# Compute diagonal zeeman corrections
def zeeman_g(B: float) -> float:
    """
    Computes the lowest order correction to the |6S1/2;4,0> state using the second order correction
    to the energy level

    Args:
        B : Magnetic field strength. Assumed to be along the Z-direction (T)

    Returns:
        shift : The energy shift in the |6S1/2;4,0> state. Converted to angular frequency (Hz)
    """
    # The matrix element between the clock states <4,0|H_B|3,0> (Hz)
    me_4_3 = N(zeeman_me_hf(B, SphericalVector([1, 0, 0]), 0, 1 / 2, 4, 0, 1 / 2, 3, 0) / hb)
    return abs(me_4_3) ** 2 / cs_clock


def zeeman_e(B: float, fe: moment, me: moment):
    """
    Computes the 1st order zeeman shift on the state |5D5/2;fe,me> using the zeeman_1o_hf() function

    Args:
        B : Magnetic Field strength. Assumed to be along the quantization axis (T)
        fe : hyperfine angular momentum quantum number. Should be int or half-int
        me : azimuthal angular momentum quantum number. Should be int or half-int

    Returns:
        shift : The energy shift in the |5D5/2;fe,me> state. Converted to angular frequency (Hz)
    """
    return N(zeeman_1o_hf(B, 2, 5 / 2, fe, me) / hb)


def zeeman_R(B: float, nr: int, mr: moment):
    """
    Computes the 1st order zeeman shift on the state |nrP3/2,mr> using the zeeman_1o_fs() and the
    diamagnetic_1o() functions on the state.

    Args:
        B: Magnetic Field strength. Assumed to be along the quantization axis (T)
        nr: principle quantum number of the Rydberg state
        mr: azimuthal quantum number of the Rydberg state. Should be int or half-int

    Returns:
        shift : The energy shift in the |nrP3/2,mr> state. Converted to angular frequency (Hz)
    """
    return N((zeeman_1o_fs(B, 1, 3 / 2, mr) + diamagnetic_1o_fs(B, nr, 1, 3 / 2, mr)) / hb)