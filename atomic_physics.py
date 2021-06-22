"""
Holds computations of internal atomic properties
"""
import numpy as np
from typing import TypeVar, List
from numpy import pi
from sympy.physics.wigner import clebsch_gordan


from SphericalTensors import SphericalVector

from basics import *

moment = TypeVar("moment", int, float)
gi_cs = -0.00039885395  # Cesium nuclear g-factor
gl_cs = 0.99999587  # Corrected electron orbital g-factor in cesium
gs = 2 + 2.31930443622e-3  # Corrected electron spin g-factor

cs_clock = 9.192631770e9  # Cesium clock frequency (Hz)

# Line-widths of relevant transitions in Cesium
gamma_5D = 1/1280e-9  # for |5D5/2> levels
# -- Computing lifetime for rydberg atoms in |nP3/2> states  --
Ts_p_3o2 = 3.2849  # From Entin 2009
delta_p_3o2 = 2.9875  # From Entin 2009
A_t, B_t, C_t, D_t = [0.038, 0.056, 1.552, 3.505]  # From Entin 2009
def ryd_lifetime_np3o2(n_eff : float, temperature: float = 0):
    """
    Compute the lifetime of a rydberg level |np3/2> at a finite temperature. If
    temperature is not specified lifetime is taken to be at 0K.
    Args:
        n_eff : Effective principle quantum number of the rydberg level. Can
            be computed as  $n - quantum_defect(n,l,j)$
        temperature : temperature of the environment of the atom in Kelvin (K).
            default is 0 K

    Returns:
        Excited state lifetime of a rydberg level
    """
    T_at_T0 = Ts_p_3o2 * n_eff ** delta_p_3o2
    if temperature == 0:
        return T_at_T0 * 1e-9
    else:
        exp_arg = 315780 * B_t / (temperature * n_eff ** C_t)
    return 1e-9/(1/T_at_T0 + A_t * 21.4 / (n_eff**D_t * (np.exp(exp_arg) - 1)))


# Quantum defects
# -- for |nP3/2> --
delta0_p_3o2 = 3.55907  # from Merkt 2016
delta2_p_3o2 = 0.375  # from Merk 2016
def defect_p_3o2(n: int) -> float:
    """
    Computes the quantum defect for a rydberg level in Cesium |nP3/2>. Uses
    defect values from Merkt 2016.
    Args:
        n : principle quantum number of rydberg level

    Returns:
        delta : quantum defect of that level taken to the next-to-highest order term
    """
    return delta0_p_3o2 + delta2_p_3o2 / (n - delta0_p_3o2) ** 2


def hf_splittings(
        a: float,
        b: float,
        i: moment,
        j: moment,
        com: bool = False
) -> np.array:
    """
    Computes the strength of hyperfine splittings between hyperfine levels in a given fine structure
    manifold, denoted by j.

    Detuning is computed relative to the lowest energy level by default. If detuning from CoM is
    desired set com arg to True.
    Args:
        a : A hyperfine coefficient (Hz)
        b : B hyperfine coefficient (Hz)
        i : nuclear spin quantum number. Should be integer or half-integer
        j : Fine-structure angular momentum quantum number for this fs level.
        com : If True, detunings are returned relative to the Center of Mass of the fine
            structure level. Otherwise detunings are returned relative to the lowest energy
            hyperfine level.

    Returns:
        list of hyperfine detunings for all hyperfine manifolds in the fine structure level.
            List is indexed by f quantum number. If an f quantum number is indexed, but does not
            correspond to a real f level, that entry in the list is set to None
            For example, if given a spin 7/2 atom (such as cesium), with and a j = 5/2 fine
            structure manifold, detunings are returned as a len(7/2+5/2 + 1= 6 + 1)
            array. Entries indexed 0 and 1 are None and the rest correspond to the actual detuning.
            The list would look like :
                [
                    None,
                    f = 1 detuning,
                    f = 2 detuning,
                    f = 3 detuning,
                    f = 4 detuning,
                    f = 5 detuning,
                    f = 6 detuning
                ]
    """
    if int(i+j) - (i+j) != 0:
        raise NotImplementedError("Currently no support for half-integer f levels")
    d_list = np.array([None] * int(i + j + 1))
    # TODO : Support for HF levels that are half-integers
    fs = np.arange(int(abs(i-j)), int(i + j) + 1)
    Kk = fs * (fs + 1) - i * (i + 1) - j * (j + 1)
    # print(dict(zip(fs, Kk)))
    dfeA = pi * a * Kk
    dfeB = b * (3/2 * Kk * (Kk + 1) + 2 * i * (i + 1) * j * (j + 1))/(4 * i * (2 * i - 1) * j * (2 * j - 1))

    if com:
        d_list[int(abs(i - j)):] = (dfeA + dfeB)
    else:
        d_list[int(abs(i - j)):] = (dfeA + dfeB) - min(dfeA + dfeB)

    return d_list


def g_fs(j: moment, lo: moment, s: moment) -> float:
    """
    Computes the gyromagnetic ratio or a state/level in the fine structure basis
    Args:
        j: Fine structure angular momentum quantum number of the state. Should be integer or
            half-integer
        lo : Orbital angular momentum quantum number of the state. Should be integer or
            half-integer
        s : Spin angular momentum quantum number of the state. Should be integer or half-integer
        gl : electron orbital g-factor
        gs : electron spin g-factor

    Returns:
        gj : SO coupled Lande g-factor
    """
    l_cont = gl_cs * (j * (j + 1) + lo * (lo + 1) - s * (s + 1))
    s_cont = gs * (j * (j + 1) - lo * (lo + 1) + s * (s + 1))
    return (l_cont + s_cont) / (2 * j * (j + 1))


def g_hf(f: moment, j: moment, lo: moment, i: moment) -> float:
    """
    Computes the gyromagnetic ratio or a state/level in the hyperfine structure basis
    Args:
        f: Hyperfine structure angular momentum quantum number of the state. Should be integer or
            half-integer
        j: Fine structure angular momentum quantum number of the state. Should be integer or
            half-integer
        lo : Orbital angular momentum quantum number of the state. Should be integer or
            half-integer
        i : Nuclear Spin angular momentum quantum number of the state. Should be integer or
            half-integer
        gl : g factor of the electron's orbital angular momentum
        gs : g factor of the electron's spin angular momentum
        gi : g factor of the nucleus's spin angular momentum

    Returns:
        gf : SO coupled Gyromagnetic ratio.
    """
    gj = g_fs(j, lo, 1/2)
    j_cont = gj * (f * (f + 1) + j * (j + 1) - i * (i + 1))
    i_cont = gi_cs * (f * (f + 1) - j * (j + 1) + i * (i + 1))
    return (j_cont + i_cont) / (2 * f * (f + 1))


def zeeman_1o_hf(Bz: float, lo: moment, j: moment, f: moment, m: moment):
    """
    Computes the zeeman shift (to first order) on a hyperfine state of Cesium
    Args:
        Bz : Magnetic field strength perturbing the atom, presumed to be along the quantization axis
        lo : orbital angular momentum quantum number of the state. Should be an integer or
            half-integer
        j : fine structure angular momentum quantum number of the state. Should be an integer or
            half-integer
        f : hyperfine angular momentum quantum number of the state. Should be an integer or
            half-integer
        m : azimuthal angular momentum quantum number of the state. Should be an integer or
            half-integer

    Returns:
        First order zeeman shift of the state (in Joules)
    """
    return mub * Bz * g_hf(f, j, lo, 7/2) * m


def zeeman_1o_fs(Bz: float, lo: moment, j: moment, m: moment) -> float:
    """
    Computes the zeeman shift (to first order) on a fine structure state of Cesium
    Args:
        Bz : Magnetic field strength perturbing the atom, along the quantization axis
        lo : orbital angular momentum quantum number of the state. Should be an integer or
            half-integer
        j : fine structure angular momentum quantum number of the state. Should be an integer or
            half-integer
        m : azimuthal angular momentum quantum number of the state. Should be an integer or
            half-integer

    Returns:
        First order zeeman shift of the state (J)
    """
    return mub * Bz * g_fs(j, lo, 1/2) * m


def diamagnetic_1o(Bz: float, n: int, lo: moment, m: moment) -> float:
    """
    Computes the diamagnetic response of an atom to a magnetic field along the quantization axis, to
    first order.

    TODO: perform the numerical integral so this works for Cesium not Hydrogen
    Args:
        Bz : Magnetic field strength perturbing the atom, along the quantization axis
        n : principle quantum number
        lo : orbital angular momentum quantum number of the state. Should be an integer or
            half-integer
        m : azimuthal angular momentum quantum number of the state. Should be an integer or
            half-integer

    Returns:
        <H_diamag> : diagonal matrix element of the diamagnetic response of an atom to a magnetic
            field. (J)
    """
    rsq = n ** 2 * (5 * n ** 2 + 1 - 3 * lo * (lo + 1)) / 2  # True for hydrogen, need cs function
    sn_lm = 2 / 3 * (1 - clebsch_gordan(lo, 2, lo, 0, 0, 0) * clebsch_gordan(lo, 2, lo, m, 0, m))

    return (mub * Bz) ** 2 / (Eh * 4) * rsq * sn_lm


def diamagnetic_1o_fs(B, n, lo, j, mj):
    """
    Computes the diamagnetic response of a state |n, lo, j, mj> to a magnetic field along the
    quantization axis.

    TODO: perform the numerical integral so this works for Cesium not Hydrogen
    Args:
        B : Field strength of the perturbing magnetic field along the quantization axis (T)
        n : principle quantum number of the state
        lo : orbital angular momentum quantum number of the state. Integer or half-integer
        j : fine structure angular momentum quantum number of the state. Integer or half-integer
        mj : azimuthal angular quantum number of the state . Integer or half-integer

    Returns:
        Correction to the state's energy due to the diamagnetic term. (Hz)
    """
    s = 0
    for ms in np.arange(-1 / 2, 1 / 2 + 1, 1):
        ml = mj - ms
        if abs(ml) > lo:
            continue
        clgd = clebsch_gordan(lo, 1 / 2, j, ml, ms, mj) ** 2
        dia = diamagnetic_1o(B, n, lo, ml)
        # print(dia / h)
        s += clgd * dia

    return s


def zeeman_me_fs(
        B: float,
        b_pol: SphericalVector,
        lo: moment,
        j: moment,
        mj: moment,
        jp: moment,
        mp: moment
) -> float:
    """
    Computes the matrix element between two states in the fine structure basis given a DC
    magnetic field.

    <a, lo, jp, mp | H_B | a, lo, j, mj>
    where H_B is the perturbing Hamiltonian due to the Zeeman effect and a abstracts away all other
    quantum numbers

    Args:
        B : magnitude of the magnetic field perturbing the atoms. (T)
        b_pol : SphericalVector describing the direction of the magnetic field.
        lo : orbital angular momentum quantum number of the states. Integer or half-integer
        j : fine structure angular momentum quantum number of the un-primed state. Integer or
            half-integer
        mj : azimuthal angular momentum quantum number of the un-primed state. Integer or
            half-integer
        jp : fine structure angular momentum quantum number of the primed state. Integer or
            half-integer
        mp : azimuthal angular momentum quantum number of the primed state. Integer or half-integer

    Returns:
        Matrix element between the states provided. (J)
    """
    s = 0
    for q in [-1, 0, 1]:
        for ms in [-1/2, 1/2]:
            ml = mj - ms
            if abs(ml) > lo:
                continue
            for msp in [-1/2, 1/2]:
                mlp = mp - msp
                if abs(mlp) > lo:
                    continue
                # print(f"lo,j,mj,ml,ms,jp,mp,mlp,msp : {lo,j,mj,ml,ms,jp,mp,mlp,msp}")
                clgd = clebsch_gordan(lo,1/2,jp,mlp,msp,mp) * clebsch_gordan(lo,1/2,j,ml,ms,mj)
                # print(f"clgd : {clgd}")
                if ml != mlp:
                    s_cont = 0
                else:
                    s_cont = gs * np.sqrt(3) / 2 * clebsch_gordan(1, 1/2, 1/2, q, ms, msp)
                if ms != msp:
                    l_cont = 0
                else:
                    l_cont = gl_cs * np.sqrt(lo * (lo + 1)) * clebsch_gordan(1, lo, lo, q, ml, mlp)
                s += b_pol[-q] * clgd * (s_cont + l_cont) * (-1) ** q
                # print(f"s : {s}")

    return B * mub * s


def zeeman_me_hf(
        B: float,
        b_pol: SphericalVector,
        lo: moment,
        j: moment,
        f: moment,
        mf: moment,
        jp: moment,
        fp: moment,
        mp: moment
) -> float:
    """
    Computes the matrix element between two states in the fine structure basis given a DC
    magnetic field.

    <a, lo, jp, fp, mp | H_B | a, lo, j, f, mf>
    where H_B is the perturbing Hamiltonian due to the Zeeman effect and a abstracts away all other
    quantum numbers

    Args:
        B : magnitude of the magnetic field perturbing the atoms. (T)
        b_pol : SphericalVector describing the direction of the magnetic field.
        lo : orbital angular momentum quantum number of the states. Integer or half-integer
        j : fine structure angular momentum quantum number of the un-primed state. Integer or
            half-integer
        f : hyperfine structure angular momentum quantum number of the un-primed state. Integer
            or half-integer
        mf : azimuthal angular momentum quantum number of the un-primed state. Integer or
            half-integer
        jp : fine structure angular momentum quantum number of the primed state. Integer or
            half-integer
        fp : hyperfine structure angular momentum quantum number of the primed state. Integer or
            half-integer
        mp : azimuthal angular momentum quantum number of the primed state. Integer or half-integer

    Returns:
        Matrix element between the states provided. (J)
    """
    i = 7/2
    s = 0
    for q in [-1, 0, 1]:
        for mI in np.arange(-7/2,7/2+1,1):
            mj = mf - mI
            if abs(mj) > j:
                continue
            for mIp in np.arange(-7/2, 7/2+1, 1):
                mjp = mp - mIp
                if abs(mjp) > jp:
                    continue
                clgd = clebsch_gordan(jp,i,fp,mjp,mIp,mp) * clebsch_gordan(j,i,f,mj,mI,mf)
                if mj != mjp:
                    I_cont = 0
                else:
                    I_cont = gi_cs * np.sqrt(i * (i + 1)) * clebsch_gordan(1, i, i, q, mI, mIp)
                if mI != mIp:
                    j_cont = 0
                else:
                    qar = [0, 0, 0]
                    qar[-q] = 1
                    j_cont = (-1)**q * zeeman_me_fs(1, SphericalVector(qar), lo, j, mj, jp, mjp)/mub
                s += b_pol[-q] * clgd * (I_cont + j_cont) * (-1) ** q
                # print(f"mj,mjp,mI,mIp = {mj,mjp,mI,mIp}")
                #print(f"q = {q}")
                #print(f"b_pol = {b_pol[-q]}")
                # print(f"clgd = {clgd}")
                #print(f"I_cont = {I_cont}")
                #print(f"j_cont = {j_cont}")
                # print(s)

    return B * mub * s

