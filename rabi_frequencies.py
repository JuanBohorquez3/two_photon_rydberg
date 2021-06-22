"""
Functions useful for computing rabi frequencies between states coupled by single fields.
"""
from basics import *
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j
from sympy import N
from typing import TypeVar, List

from SphericalTensors import SphericalVector


"""
meant to represent angular momenta quantum numbers. Should be filled with integer or half-integer 
values 
"""
moment = TypeVar("moment", int, float)


# TODO:  Write Latex document to detail derivations or arguments for these functions
def dipole_rme_hf(
        d_fs: complex,
        j: moment,
        f: moment,
        jp: moment,
        fp: moment,
        i: moment,
) -> complex:
    """
    Computes the reduced dipole matrix element between two hyperfine levels, given the reduced
    dipole matrix element between those levels in the fine structure basis
    Args:
        d_fs : reduced dipole matrix element between the fine structure levels j and jp (Cm)
            <ap, jp||er||a, j>
        j : fine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        f : hyperfine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        jp : fine structure angular momentum quantum number for the primed state. Should be
            integer or half-integer
        fp : hyperfine structure angular momentum quantum number for the primed state. Should be
            integer or half-integer
        i: nuclear spin quantum number. Should be integer or half-integer

    Returns:
        d_hf : reduced dipole matrix element between the hyperfine levels |a, j,f> and
            |ap, jp,fp> (Cm).
            Equivalent to <ap, jp, fp|| er ||a, j, f>
    """
    return (-1)**(1+i+jp+fp)*np.sqrt((2*f+1)*(2*fp+1))*N(wigner_6j(f, 1, fp, jp, i, j))*d_fs


def dipole_rme_fs(
        d_l: complex,
        li: moment,
        j: moment,
        lp: moment,
        jp: moment,
        s: moment
) -> complex:
    """
    Computes the reduced dipole matrix element between two levels in the fine structure basis given
        the reduced matrix element between those levels in the orbital basis.
    Args:
        d_l : reduced matrix element between orbital levels li, and lp (Cm). <ap, lp || er ||a, li>
        li : orbital angular momentum quantum number for the initial state. Should be integer or
            half-integer
        j : fine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        lp : orbital angular momentum quantum number for the primed state. Should be integer or
            half-integer
        jp : fine structure angular momentum quantum number for the primed state. Should be
            integer or half-integer
        s : electron spin momentum quantum number. Should be integer or half-integer

    Returns:
        d_fs : reduced matrix elements between the fine structure level |a, li, j> and
        |ap, lp, jp> (Cm). Equivalent to <ap, lp, jp|| er ||a, li, j>.
    """
    return (-1) ** (1+j+lp+s) * np.sqrt((2*j+1)*(2*jp+1)) * N(wigner_6j(li, s, j, jp, 1, lp)) * d_l


def quadrupole_rme_hf(
        q_fs: complex,
        j: moment,
        f: moment,
        jp: moment,
        fp: moment,
        i: moment,
) -> complex:
    """
    Computes the reduced quadrupole matrix element between two hyperfine levels, given the reduced
    dipole matrix element between those levels in the fine structure basis
    Args:
        q_fs : reduced quadrupole matrix element between the fine structure levels j and jp (Cm**2)
            <ap, jp|| e C_2/sqrt(15) ||a, j>
        j : fine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        f : hyperfine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        jp : fine structure angular momentum quantum number for the primed state. Should be
            integer or half-integer
        fp : hyperfine structure angular momentum quantum number for the primed state. Should be
            integer or half-integer
        i: nuclear spin quantum number. Should be integer or half-integer

    Returns:
        d_hf : reduced dipole matrix element between the hyperfine levels |a, j,f> and
            |ap, jp,fp> (Cm).
            Equivalent to <ap, jp, fp|| er ||a, j, f>
    """
    return (-1)**(f+jp+i+2)*np.sqrt((2 * f + 1)*(2 * fp + 1))*N(wigner_6j(j, i, f, fp, 2, jp))*q_fs


def quadrupole_rme_fs(
        q_l: complex,
        li: moment,
        j: moment,
        lp: moment,
        jp: moment,
        s: moment
) -> complex:
    """
    Computes the reduced quadrupole matrix element between two levels in the fine structure basis
    given
        the reduced matrix element between those levels in the orbital basis.
    Args:
        q_l : reduced matrix element between orbital levels li, and lp (Cm**2)
            <ap, lp || e C_2/sqrt(15) ||a, li>
        li : orbital angular momentum quantum number for the initial state. Should be integer or
            half-integer
        j : fine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        lp : orbital angular momentum quantum number for the primed state. Should be integer or
            half-integer
        jp : fine structure angular momentum quantum number for the primed state. Should be
            integer or half-integer
        s : electron spin momentum quantum number. Should be integer or half-integer

    Returns:
        q_fs : reduced matrix elements between the fine structure level |a, li, j> and
        |ap, lp, jp> (Cm). Equivalent to <ap, lp, jp|| er ||a, li, j>.
    """
    return (-1)**(j+lp+s+2)*np.sqrt((2 * j + 1)*(2 * jp + 1))*N(wigner_6j(li, s, j, jp, 2, lp))*q_l


def dipole_rabi_frequency(
        electric_field: complex,
        q: SphericalVector,
        d_rme: complex,
        j: moment,
        m: moment,
        jp: moment,
        mp: moment,
) -> complex:
    """
    Computes rabi frequency between two states zeeman states given electric field strength and
    the reduced dipole moment between them.

    Our two states can be described as |a,j,m> and |ap,jp,mp> where p represents primed states,
    and a/ap correspond to unaccounted for quantum numbers between the states.

    The returned value is given by the Wigner–Eckart theorem.

    rabi_frequency =
        electric_field/hb * c_a * clebsch_gordan(1, jp, j, q_a, mp, m) * d / sqrt(2*jp+1)

    where:
        hb = reduced planck constant, as provided in constants.py
        c_a = weight of polarization that is allowed for dipole coupling between m and mp
        q_a = polarization that is allowed for dipole coupling between m and mp
        clebsch_gordan = function from sympy.physics.wigner

    Args:
        electric_field : electric field amplitude of the oscillating field coupling two states (V/m)
        q : length three list describing the relative polarization of the oscillating field in the
            spherical basis Components should be indexed [e_0, e_+1, e_-1].
        d_rme : reduced dipole matrix element between the states (Cm)
        j : angular momentum quantum number for initial state. Int or half-int.
        m : zeeman state quantum number for initial state. Int or half-int.
        jp : angular momentum quantum number for primed state. Int or half-int.
        mp : zeeman state quantum number for primed state. Int or half-int.

    Returns:
        rabi frequency coupling |a,j,m> and |ap,jp,mp> in Hz (assuming values passed in were in
            correct units)
    """
    q_a = mp-m  # dipole allowed field polarization
    if abs(q_a) > 1:  # dipole only allows change of 1
        return 0

    # proportion of field in that polarization state
    c_a = q[int(q_a)]
    # print(f"m-mp:{q_a},polarization_array:{q},c_a:{c_a}")
    return electric_field / hb * c_a * N(clebsch_gordan(1, j, jp, q_a, m, mp)) * d_rme / \
           np.sqrt(2 * jp + 1)


def quadrupole_rabi_frequency(
        electric_field: complex,
        frequency: float,
        q_ar: SphericalVector,
        k_ar: SphericalVector,
        q_rme: complex,
        j: moment,
        m: moment,
        jp: moment,
        mp: moment
) -> complex:
    """
    Computes the rabi frequency between two zeeman states given the electric field strength, reduced
    quadrupole moment between them, the field polarization, and the field k-vector polarization.

    The two zeeman states can be denoted as |a, j, m> and |ap, jp, mp>, where p denotes primed
    states, and a/ap abstract away all unaccounted for quantum numbers that describe the states.

    Args:
        electric_field : electric field amplitude of the oscillating field coupling two states (V/m)
        frequency : radial oscillation frequency of the field coupling the two states (Hz)
        q_ar : length three list describing the relative polarization of the oscillating field in
            the spherical basis. Components should be indexed [e_0, e_+1, e_-1].
        k_ar : length three list describing the relative polarizatio of the oscillating field's
            k-vector in the spherical basis. Components should be indexed [e_0, e_+1, e_-1]
        q_rme : reduced quadrupole matrix element between the states (Cm^2)
        j : angular momentum quantum number for initial state. Int or half-int.
        m : zeeman state quantum number for initial state. Int or half-int.
        jp : angular momentum quantum number for primed state. Int or half-int.
        mp : zeeman state quantum number for primed state. Int or half-int.

    Returns:

    """
    def m_q(qs: SphericalVector, ks: SphericalVector, q: int) -> float:
        ret = 0
        for moo in range(-1, 2):  # supposed to be $\mu$
            for nuu in range(-1, 2):  # supposed to be $\nu$
                ret += N(clebsch_gordan(1, 1, 2, moo, nuu, -q))*ks[moo]*qs[nuu]
        return (-1)**q * np.sqrt(10) * ret
    pre = 1j * k_mag(frequency*2*pi) * electric_field * q_rme / (hb * np.sqrt(2 * jp + 1))
    # print(pre)
    # print(electric_field)
    # print(f"f = {frequency*1e-12}THz")
    # print(f"k = {k_mag(frequency*2*pi)}m^-1")

    m_sum = sum([m_q(q_ar, k_ar, q)*N(clebsch_gordan(2, j, jp, q, m, mp)) for q in range(-2, 3)])
    return complex(pre * m_sum)


def dipole_hf_to_fs(
        electric_field: complex,
        q_ar: SphericalVector,
        d_rme: complex,
        j: moment,
        f: moment,
        mf: moment,
        jp: moment,
        mp: moment,
        i: moment
) -> complex:
    """
    Computes rabi frequency for a dipole transition between two zeeman states, the lower
    (un-primed) state being defined in the hyperfine basis and the upper (primed) state being
    defined in the fine structure basis.

    This computation is particularly useful in the case of exciting to Rydberg states, where the
    strength hyperfine interactions become much smaller than the linewidths of the fine structure
    levels.

    The states in this function are described as |a, f, mf> -> |ap, jp, mp> where p represents
    primed states, and a/ap abstract away all unaccounted-for quantum numbers.
    Args:
        electric_field : electric field strength of the oscillating field coupling the two states
        q_ar : SphericalVector object describing the field polarization
        d_rme : reduced dipole matrix element between the two FINE-STRUCTURE levels
        j : fine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        f : hyperfine structure angular momentum quantum number for the initial state. Should be
            integer or half-integer
        mf : zeeman state quantum number (in the hyperfine basis) for the initial state. Should be
            integer or half-integer
        jp : fine structure angular momentum quantum number for the final state. Should be
            integer of half-integer
        mp : zeeman state quantum number (in the fine structure basis) for the final state.
            Should be integer or half-integer
        i : nuclear spin quantum number. Should be integer or half-integer

    Returns:
        rabi frequency coupling |a,I,j,f,mf> to |ap,j,mj;I>
    """

    s = 0
    for q in [-1, 0, 1]:
        #print(q, q_ar[q])
        for fp in np.arange(abs(i-jp), i+jp+1):
            mfp = q+mf
            if abs(mfp) > fp:
                # print(f"q = {q} mf = {mf}, fp = {fp} mfp = {mfp}")
                s += 0
                continue
            c1 = clebsch_gordan(jp, i, fp, mp, mfp-mp, mfp)
            c2 = clebsch_gordan(1, f, fp, q, mf, mfp)
            six = wigner_6j(j, i, f, fp, 1, jp)
            cont = c1 * c2 * six * q_ar[q] * (-1) ** (1 + i + jp + fp)
            # if(q_ar[q] != 0):
                # print(f"fr = {fp} mfr = {mfp} c1 = {c1} c2 = {c2}")
            s += cont
    return electric_field * N(s) * np.sqrt(2 * f + 1) * d_rme / hb