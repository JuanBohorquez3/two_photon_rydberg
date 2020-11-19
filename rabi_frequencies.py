"""
Functions useful for computing rabi frequencies between states coupled by single fields.
"""
from basics import *
from sympy.physics.wigner import clebsch_gordan


def dipole_rabi_frequency(
        electric_field: float,
        q: list,
        d: float,
        j: int,
        m: int,
        jp: int,
        mp: int,
):
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
        electric_field : electric field amplitude of field coupling two states (V/m)
        q : length three list describing the relative polarization of the light in the spherical
            basis [e_+1, e_0, e_-1]
        d : reduced dipole matrix element between the states (m)
        j : angular momentum quantum number for initial state
        m : zeeman state quantum number for initial state
        jp : angular momentum quantum number for other state
        mp : zeeman state quantum number for other state

    Returns:
        rabi frequency coupling |a,j,m> and |ap,jp,mp> in Hz (assuming values passed in were in
            correct units)
    """
    q_a = m-mp  # dipole allowed field polarization
    if abs(q_a) > 1:  # dipole only allows change of 1
        return 0

    # proportion of field in that polarization state
    try:
        c_a = np.array(q)[np.where(q == q_a)][0]
    except IndexError:
        return 0

    return \
        electric_field / hb * c_a * clebsch_gordan(1, jp, j, q_a, mp, m) * d / np.sqrt(2 * jp + 1)
