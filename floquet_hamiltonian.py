from typing import List, Dict, Union
import numpy as np
from arc import *
from time import time
import os

from angular_momentum import d_jmm
from utility import *
from basics import *
pi = np.pi

RESULTS_FOLDER = "Shirley-Floquet_results"

FORMATTED = {
    "field_omega": "om-{:.0f}",
    "Eac"        : "eac-{:.0f}",
    "ellipticity": "eps-{:.3f}",
    "Edc"        : "edc-{:.0f}",
    "theta"      : "th-{:.3f}"
}

LONGER = {
    "om": "field_omega",
    "eac": "Eac",
    "eps": "ellipticity",
    "edc": "Edc",
    "th": "theta"
}

"""
Functions that are useful for Shirley-Floquet analysis of a periodic system
"""
# --------------------------------------------------------------------------------------------------
# == Functions that build relevant matrices ==
# --------------------------------------------------------------------------------------------------


def build_H0(
        basis: List[RydStateFS],
        t_state: RydStateFS = None,
        offset_zeeman=0,
        j_split=0
) -> np.ndarray:
    """
    Build Bare atomic hamiltonian

    State energies computed using arc. Computed for states in cesium.
    Args:
        basis: list of states making up the basis. Each item should be a RydStateFS object
        t_state: "target" state, this state is the state we care about. This state defines 0 energy
            (all energies measured against it). If None, energies reported are relative to
            ionization energy
        offset_zeeman: Offset due to small ficticious zeeman field. default = 0 (radians/s)
    Returns:
        H0: Diagonal hamiltonian of the unperturbed cesium atom. Diagonal entries are unperturbed
            energies of basis states. energies are reported relative to the energy of t_state (or
            ionization energy), energies reported in radial frequency (radians/s)
    """
    if t_state is not None:
        H0 = np.diag(
            np.array(
                [detuning(state, t_state, 0) + offset_zeeman * state["mj"] + j_split * state.j
                 for state in basis],
                dtype=complex
            )
        )
    else:
        H0 = np.diag(
            np.array(
                [state.get_energy() * e / hb + offset_zeeman * state["mj"] + j_split * state.j for
                 state in basis],
                dtype=complex
            )
        )
    return H0


def dipoles(basis: list, qs=None):
    """
    Builds the matrix of dipole moments between basis states. Assumes dipole field is pi-polarized.
    Dipole moments computed using arc. Computed for rydberg states in cesium.

    Args:
        basis: list of states that make up our basis
        qs: list of proportions of polarization states in driving field. For dc fields should always
         be [1,0,0] (rotations should be accounted for later). If qs is not normalized it's
         normalized within the function. indeces are [pi, sigma_plus, sigma_minus]
    Returns:
        d : Matrix of dipole moments (divided by hbar) between basis states (radians/s/(V/m))
    """
    if qs is None:
        qs = [1, 0, 0]
    qs = np.array(qs) / sum([abs(q)**2 for q in qs])  # normalize qs

    d = np.zeros((len(basis), len(basis)), dtype=complex)
    for q in [-1, 0, 1]:
        for i, state in enumerate(basis):
            for j, statep in enumerate(basis):
                if abs(statep.l - state.l) != 1:
                    continue
                if abs(statep.j - state.j) > 1:
                    continue
                if statep.get_energy() > state.get_energy():
                    continue
                # print(f"Computing Matrix Element {state.ket()} x {statep.ket()}")
                d[i, j] += qs[q] * state.atom.getDipoleMatrixElement(
                    statep.n,
                    statep.l,
                    statep.j,
                    statep["mj"],
                    state.n,
                    state.l,
                    state.j,
                    state["mj"],
                    q
                ) * ao * e / hb
    return d + np.conjugate(d.T)


def build_floquet(
        basis: List[RydStateFS],
        H0: np.ndarray,
        Eac: float,
        ellipticity: float,
        field_omega: float,
        Edc: float,
        theta: float,
        n_max: int,
        dipoles_z: np.ndarray = None,
        dipoles_ac: np.ndarray = None,
        **kwargs
) -> np.ndarray:
    """
    Build Floquet hamiltonian in Shirley-Floquet formalism. For use with a single harmonic driving
        field
    Args:
        basis: Lists of states in this basis
        H0: Hamiltonian describing bare atomic system. Should be diagonal in basis provided. Matrix 
            elements should be reported in Radial Frequency (radians/s)
        Eac: Electric field strength of AC field (V/m)
        ellipticity: parameter describing how elliptical ac field polarization is. AC polarization
            is then described as e_AC = e_pi*sqrt(1-ellipticity) + e_sigma_+*sqrt(ellipticity)
        field_omega: radial frequency of driving field. (radians/s)
        Edc: Electric field strength of DC field (V/m)
        theta: angle between the quantization axis and the direction of the DC electric field. 
        (radians)
        n_max: maximum number of floquet photons to consider in floquet Hamiltonian

    Returns:
        Floquet Hamiltonian constructed in the Shirley-Floquet formalism. Matrix elements reported
            in radial frequency (radians/s).
    """

    def Hdc_n(Ham, n, omega_d):
        '''Build Diagonal block for Floquet hamiltonian'''
        return Ham + np.diag(n * omega_d * np.ones(Ham.shape[0]))
    if dipoles_z is None:
        ts = time()
        dipoles_z = dipoles(basis, [1, 0, 0])
        print(f"dipoles_Z built in {time()-ts}")
    ts = time()
    little_d = build_little_d(theta, basis)
    print(f"little_d built in {time()-ts}")
    ts = time()
    Hdc = H0 + Edc*np.dot(np.dot(little_d.T,dipoles_z),little_d)
    print(f"Hdc built in {time()-ts}")
    if dipoles_ac is None:
        ts = time()
        dipoles_ac = dipoles(basis, [np.sqrt(1-ellipticity), np.sqrt(ellipticity), 0])
        print(f"dipoles_ac built in {time()-ts}")
    ts = time()
    Hfloquet = np.zeros([dim * (2 * n_max + 1) for dim in H0.shape], dtype=complex)
    for i, ni in enumerate(range(-n_max, n_max + 1)):
        for j, nj in enumerate(range(-n_max, n_max + 1)):
            if i == j:
                block = Hdc_n(Hdc, ni, field_omega)
            elif abs(i - j) == 1:
                block = Eac * dipoles_ac / 2
            else:
                continue
            Hfloquet[len(basis) * i:len(basis) * (i + 1), len(basis) * j:len(basis) * (j + 1)] = \
                block
    print(f"Hfloquet put together in {time()-ts}")
    return Hfloquet


def build_little_d(beta: float, basis: List[RydStateFS]) -> np.ndarray:
    """
    Builds the little rotation matrix d from matrix elements d_jmm

    Produces a rotation by angle beta about the y-axis.
    Use with z-rotations to produce arbitrary rotations.
    Args:
        beta : angle of rotation (radians)
        basis : basis describing the system's Hilbert space

    Returns:
        little_d : an NxN matrix that produces rotations about the y-axis when applied to states
            within the Hilbert space defined by basis. N = len(basis). Matrix elements are floats.
    """
    def check_qn(stat: RydStateFS, statp: RydStateFS) -> bool:
        """
        Checks that all quantum numbers in state match that of statep, excluding quantum numbers
        labeled "mj". This quantum number is considered to be the azimuthal quantum number of the
        states.
        Args:
            stat : unprimed state to be considered
            statp : primed state to be considered
        Returns:
            True if all non-"mj" quantum numbers are the same for both states
            False otherwise
        """
        qns = {label: value for label, value in stat.quantum_numbers().items() if label != "mj"}
        qnsp = {label: value for label, value in statp.quantum_numbers().items() if label != "mj"}
        return qns == qnsp

    ds = np.zeros((len(basis), len(basis)), dtype=float)
    for i, state in enumerate(basis):
        for j, statep in enumerate(basis):
            if check_qn(state, statep):
                ds[i, j] = d_jmm(beta, state.j, state["mj"], statep["mj"])
    return ds

# --------------------------------------------------------------------------------------------------
# == Functions which act on Shirley-Floquet matrices ==
# --------------------------------------------------------------------------------------------------


def floquet_diag(
        basis: List[RydStateFS],
        H0: np.ndarray,
        Eac: float,
        ellipticity: float,
        field_omega: float,
        Edc: float,
        theta: float,
        n_max: int,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds and diagonalizes the shirley-floquet Hamiltonian corresponding to a periodically driven
    system described by the arguments provided

    System is an atom irradiated by an rf field (the dressing field) with frequency field_omega and
    electric field
    strength Eac. The field can have a non-zero ellipticity provided by the ellipticity paramter.
    The field polarization is described as
    e_ac = sqrt(1-ellipticity)e_pi + sqrt(ellipticity)e_+
    The quantization axis is taken to be along the linear part of the AC field's polarization.

    basis is composed of atomic states that are to be considered in this computation. H0 is the
    Hamiltonian describing the atomic system.

    The atom is further perturbed by a DC electric field with strength Edc, and a direction that
    is at an angle theta with respect to the quantization axis defined above.

    n_max represent the number of fourier components of the dressing field to take into
    consideration when creating the floquet hamiltonian
    Args:
        basis : List of undressed atomic states to consider when creating the floquet hamiltonian
        H0 : Hamiltonian describing unperturbed atomic system. Expected to be diagonal, indexing
            should be the same as that of basis. Matrix elements should be reported in radial
            frequency (radians/s)
        Eac : Peak Electric field strength of the dressing field at the atoms (V/m)
        ellipticity : ellipticity of the dressing field at the atoms. Field polarization is
            described as: e_ac = sqrt(1-ellipticity)e_pi + sqrt(ellipticity)e_+
        field_omega : oscillation frequency of the dressing field in radial frequency. (radians/s)
        Edc : Electric field strength of the DC field at the atoms (V/m)
        theta : angle between the DC electric field and the quantization axis (radians)
        n_max : maximum number of fourier components of the dressing field to consider in the
            creation of the Floquet hamiltonian

    Returns:
        eigenvalues, eigenvectors :
            eigenvalues : list of eigenvalues of the floquet hamiltonian in no particular order.
                eigenvalues correspond to the energy of each eigenstate in eigenstates
                eigenvalues.shape = ( (2*n_max+1)*len(basis), ). (radians/s)
            eigenvectors : list of eigenvectors of the floquet hamiltonian in no particular order.
                eigenvectors correspond to the basis states of Hf that diagonalize it.
                eigenvectors are normalized. indexing is the same as that of eigenvalues (thus
                the i-th eigenvector will correspond to a state with the energy of the i-th
                eigenvalue. Indexing is such that the i-th eigenvalue will be eigenvalues[i,:].
                eigenvectors.shape = ( (2*n_max+1)*len(basis), (2*n_max+1)*len(basis) ).
    """

    t_start = time()
    Hfloquet = build_floquet(basis, H0, Eac, ellipticity, field_omega, Edc, theta, n_max, **kwargs)
    print(f"Floquet matrix built after {time()-t_start} s")

    t_start = time()
    print(f"Begin Diagonalizing")
    eigenvalues, eigenvectors = np.linalg.eig(Hfloquet)
    print(f"Diagonalized Hfloquet for with dims {Hfloquet.shape}")
    print(f"AC parameters Eac {Eac}, ellipticity {ellipticity}, field_omega {field_omega}")
    print(f"DC parameters Edc {Edc}, theta {theta}")
    print(f"Diagonalized after time: {time() - t_start}s")
    t_start = time()
    eigenvectors = eigenvectors.T
    for j, ev in enumerate(eigenvectors):
        ev = ev / (np.absolute(ev) ** 2).sum()
        eigenvectors[j] = ev
    print(f"Normalization Complete after {time()-t_start} s")
    return eigenvalues, eigenvectors


def floquet_rearrange(
        old_vectors: np.ndarray,
        new_vectors: np.ndarray,
        ips_summer: np.ndarray,
        starts: List
) -> np.ndarray:
    """
    Compares the eigenvectors in old_vectors to those in new_vectors. Determines how best to
    rearrange new_vectors, returns the indeces of these arrays in an order that will rearrange
    new_vectors to maximize overlap with old_vectors.

    Example usage:
        flen = (2*n_max+1)*len(basis)
        all_vectors = np.zeros((flen,flen,2),dtype=complex)
        all_energies = np.zeros((flen,2),dtype=float)
        old_energies, old_vectors = floquet_diag(basis,H0,n_max=n_max,**fields)
        all_vectors[...,0] = old_vectors
        all_energies[:,0] = old_energies
        fields[Eac] = fields[Eac]+2  # V/m
        new_energies, new_vectors = floquet_diag(basis,H0,n_max=n_max,**fields)
        inds = floquet_rearrange(basis, levels, old_vectors, new_vectors)
        all_vectors[inds[:,1], :, 1] = new_vectors[inds[:,0], :]
        all_energies[inds[:, 1], 1] = new_energies[ind[:, 0]]

    Args:
        old_vectors : eigenstates of previous iteration, to which new_vectors will be compared.
            rearrangement happens
        new_vectors : eigenstates of new iteration which will be rearranged
        ips_summer : matrix that compresses inner products to population within an energy level
        starts : list of the indices where each rydberg level begins within the basis
    Returns:
        inds : a 2D array of indeces that map the eigenvectors in new_vectors to their most similar
            eigenvectors in old_vectors. Similarity is measured as the square of the inner
            product between vectors.
    """
    # The easy way. Find all overlaps of new_vectors with old_vectors. For each matrix of
    # overlaps find all indices that indicate >50% overlap. Use those indices to map new_vectors
    # to old_vectors
    ips = np.abs(np.dot(np.conj(old_vectors), new_vectors))**2
    inds = np.argwhere(ips > 0.5)

    # check that all basis states in both the old_vectors and new_vectors are represented
    check0 = all([a in inds[:, 0] for a in range(old_vectors.shape[0])])
    check1 = all([a in inds[:, 1] for a in range(old_vectors.shape[0])])
    if check0 and check1:
        return inds

    # If we cannot represent all basis states the easy way, first find energy levels
    # that contain >50% of each new_eigenvector population. Then we sort the eigenvectors that
    # are within that level by maximizing the inner products with each band in the level.
    ips_levels = np.dot(ips, ips_summer)
    inds_l = np.argwhere(ips_levels > 0.5)
    if not all(a in inds_l[:, 0] for a in range(old_vectors.shape[0])):
        raise RuntimeError(f"Failed to represent all basis states\n{inds_l}")
    for j in range(ips_levels.shape[1]):
        inds_j = np.argwhere(inds_l[1] == j)




def floquet_loop(
        basis: List[RydStateFS],
        H0: np.ndarray,
        Eac: float,
        ellipticity: float,
        field_omega: float,
        Edc: float,
        theta: float,
        n_max: int,
        varied: Tuple[str, np.ndarray] = None,
        energy_bands: bool = False,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    produces, then diagonalizes a floquet hamiltonian at a variety of different system parameters.

    independent variable is defined in parameter varied, it overwrites the value of the steady state

    eigenvectors and eigenvalues are organized into a large ndarray that allows the user to track
    eigenvalues as bands in the resulting floquet spectrum as the independent variable is varied.
    This is done by having the eigenvalues at each value of the independent variable be lined up
    to maximize the overlap of each indexed eigenvector with the eigenvector for the previous
    independent variable value.

    Args:
        basis : List of undressed atomic states to consider when creating the floquet hamiltonian
        H0 : Hamiltonian describing unperturbed atomic system. Expected to be diagonal, indexing
            should be the same as that of basis. Matrix elements should be reported in radial
            frequency (radians/s)
        Eac : Peak Electric field strength of the dressing field at the atoms (V/m)
        ellipticity : ellipticity of the dressing field at the atoms. Field polarization is
            described as: e_ac = sqrt(1-ellipticity)e_pi + sqrt(ellipticity)e_+
        field_omega : oscillation frequency of the dressing field in radial frequency. (radians/s)
        Edc : Electric field strength of the DC field at the atoms (V/m)
        theta : angle between the DC electric field and the quantization axis (radians)
        n_max : maximum number of fourier components of the dressing field to consider in the
            creation of the Floquet hamiltonian
        varied : (name of independent variable, values taken by independent variable), name must
            be one of the following ["Edc", "Eac", "ellipticity", "theta"]. values must be 1D array
            of floats. This likely will only work if values start from a small (preferably 0) value
            and make their way up to the maximal value.
        energy_bands : If true, sort bands within a fine structure level (eg |52P3/2;-1>) by energy,
            otherwise bands are sorted by m_j
    Returns:
        energies, eigenvectors: ndarrays containing the values of the system's energies and
            eigenvectors at each value of varied[1]
            energies.shape = ( (2*n_max+1)len(basis), len(varied[1] )
            eigenvectors.shape = ( (2*n_max+1)len(basis), (2*n_max+1)len(basis), len(varied[1] )
    """
    if type(varied[0]) is not str:
        raise TypeError(
            "first entry in varied parameter must be a string indicating which system parameter "
            "is an independent variable")

    flen = len(basis)*(2*n_max+1)
    print(flen)
    energies = np.ones((flen, len(varied[1])), dtype=float)
    eigenstates = np.zeros((flen, flen, len(varied[1])), dtype=complex)

    levels = implied_levels(basis)
    print("Levels :\n")
    basis_print(levels)
    starts = level_starts(levels)
    ips_summer = level_projector(basis, levels, n_max)

    dipoles_z = dipoles(basis, qs=[1, 0, 0])
    if varied[0] != "ellipticity":
        # dipoles_ac = dipoles(basis, qs=[np.sqrt(1-ellipticity), np.sqrt(ellipticity), 0])
        dipoles_ac = dipoles(basis, qs=[
            np.sqrt(1-ellipticity),
            -np.sqrt(ellipticity)/np.sqrt(2),
            -np.sqrt(ellipticity)/np.sqrt(2)
        ])
    else:
        dipoles_ac = None
    for i, ival in enumerate(varied[1]):
        if varied[0] == "Edc":
            Edc = ival
        elif varied[0] == "Eac":
            Eac = ival
        elif varied[0] == "ellipticity":
            ellipticity = ival
        elif varied[0] == "theta":
            theta = ival
        else:
            raise ValueError(f"independent variable {varied[0]} not recognized")

        t_start = time()
        eigenvalues, eigenvectors = floquet_diag(
            basis,
            H0,
            Eac,
            ellipticity,
            field_omega,
            Edc,
            theta,
            n_max,
            **{"dipoles_z": dipoles_z, "dipoles_ac": dipoles_ac}
        )

        print(f"floquet_diag call completed in {time()-t_start}s")
        print(f"Diagonalization complete for independent variable entry, value {i}, {ival}")

        # re-arrange eigenvalues and eigenvectors
        t_start = time()
        # first entry is special
        if i == 0:
            # compute overlaps wrt unperturbed eigenstates
            ips = np.abs(eigenvectors)**2
            # sum over all zeeman states in each level
            ips_levels = np.dot(ips, ips_summer)
            # sum over all level in each fourier sub basis
            used_inds = []
            troublesome_level = RydStateFS(51, 2, 5/2)
            for j, level in enumerate(levels):
                for k, n in enumerate(range(-n_max, n_max+1)):
                    print(f"finding good eigenvectors for |level, n> = |{level.ket()},{n}>")
                    # eigenvectors that have >50% population in this level
                    thrsh = 0.5
                    inds_l = np.argwhere(ips_levels[:, j+k*len(levels)] > thrsh)
                    # if the 50% threshold is too high to accommodate all m levels, lower the
                    # threshold incrementally
                    while len(inds_l) < 2*level.j+1:
                        thrsh *= 0.95
                        print(f"expansion required, threshold reduced to {thrsh}")
                        inds_l = np.argwhere(ips_levels[:, j + k * len(levels)] > thrsh)

                    if energy_bands:
                        # now check if there are too many states that passed threshold, if so,
                        # choose only the states with the greatest overlap
                        if len(inds_l) > 2*level.j+1:
                            ovlps = np.argsort(ips_levels[inds_l, j + k * len(levels)],axis=0)[::-1]
                            print(ovlps)
                            print(int(2*level.j+1))
                            inds_l = inds_l[ovlps[:int(2*level.j+1), 0]]
                        # sort the eigenvectors by energy level
                        nrgs = np.argsort(np.real(eigenvalues)[inds_l], 0)
                        for index in inds_l:
                            if index in used_inds:
                                print(f"WARNING: Index {index} has already been used")
                        used_inds.extend(inds_l)
                        # Map those energies to the zeeman states within the level
                        strt = starts[j]+k*len(basis)
                        stp = starts[j+1]+k*len(basis)
                        for ind, eind in zip(range(strt, stp), nrgs):
                            eigenstates[ind, :, 0] = eigenvectors[inds_l[eind][0][0], :]
                            energies[ind, 0] = eigenvalues[inds_l[eind][0][0]]
                    else:
                        # find the band that has the greatest overlap with each mj level
                        strt = starts[j]+k*len(basis)
                        for a in range(int(2*level.j + 1)):
                            # print(f"m = {-level.j + a}")
                            # print(f"sub_ips = {ips[inds_l, strt + a]}")
                            ev_ind = np.argmax(ips[inds_l, strt + a])
                            if inds_l[ev_ind, 0] in used_inds:
                                print(f"WARNING: index {inds_l[ev_ind, 0]} has been used")
                            print(f"energy = "
                                  f"{eigenvalues[inds_l[ev_ind,0]]*1e-9/(2*pi)}GHz")
                            ev_list = eigenvectors[inds_l[ev_ind,0], :]
                            # ev_list[np.argwhere(np.abs(ev_list)**2 < 1e-2)] = np.NaN
                            ev_str = ""
                            for jj, m in enumerate(range(-n_max,n_max+1)):
                                for ii, state in enumerate(basis):
                                    mind = ii + jj*len(basis)
                                    if np.abs(ev_list[mind])**2 < 1e-2:
                                        continue
                                    st_ket = state.ket()[:-1] + f";{m}>"
                                    ev_str += f"({abs(ev_list[mind])**2:.2f})^1/2{st_ket} + "
                            print("eigenstate: \n" + ev_str[:-2])
                            used_inds.append(inds_l[ev_ind, 0])
                            # print(f"new inds = {inds_l[ev_ind,0]}")
                            # print(f"New energy = {eigenvalues[inds_l[ev_ind, 0]]}")
                            eigenstates[strt + a, :, 0] = eigenvectors[inds_l[ev_ind, 0], :]
                            energies[strt + a, 0] = np.real(eigenvalues[inds_l[ev_ind, 0]])
        else:
            # try doing this the easy way. Find all overlaps of the computed eigenstates with the
            # previous iteration's eigenstates. For each eigenstate find the previous eigenstate
            # that has >50% overlap. Use those indeces as a map for current eigenstates to previous
            # eigenstates
            ips = np.abs(np.dot(np.conj(eigenvectors), eigenstates[..., i-1].T))**2
            inds = np.argwhere(ips > 0.5)

            # check that all basis states are represented in each column of inds:
            check0 = all([a in inds[:, 0] for a in range(flen)])
            check1 = all([a in inds[:, 1] for a in range(flen)])
            if check0 and check1:
                # for ind in inds:
                #     eigenstates[ind[1], :, i] = eigenvectors[ind[0], :]
                #     energies[ind[1], i] = eigenvalues[ind[0]]
                eigenstates[inds[:, 1], :, i] = eigenvectors[inds[:, 0], :]
                energies[inds[:, 1], i] = eigenvalues[inds[:, 0]]
            else:
                # if the easy way fails, we find the eigenvectors that maximize overlap with
                # previous eigenvectors by first finding which levels have >50% population, then
                # finding which band within that level has the greatest overlap with the
                # eigenvectors. We then check that all states in the basis are represented by the
                # resulting index list
                print(inds)
                print(ips.shape, ips)
                ips_levels = np.dot(ips, ips_summer)
                inds_l = np.argwhere(ips_levels > 0.5)
                if all([a in inds_l[:, 0] for a in range(flen)]):
                    for ind_l in inds_l:
                        lvl = ind_l[1]%len(levels)
                        n_ind = int(ind_l[1]/len(levels))
                        strt = starts[lvl] + n_ind * len(basis)
                        stp = starts[lvl + 1] + n_ind * len(basis)
                        sub_ips = ips[ind_l[0],strt: stp]
                        sub_ips = sub_ips/sub_ips.sum()  # normalize
                        act_ind = np.argwhere(sub_ips == max(sub_ips))
                        ind = [ind_l[0], strt + act_ind]
                        eigenstates[ind[1], :, i] = eigenvectors[ind[0], :]
                        energies[ind[1], i] = eigenvalues[ind[0]]
                else:
                    return energies, eigenstates, ips, inds, ips_levels, inds_l
                    # raise RuntimeError(
                      #   f"Failed to represent all basis states\ncut\n{inds_l}\ncut\n{ips_levels}")
        print(f"re-arrangement done after time {time()-t_start}s")

    return energies, eigenstates

def floquet_loop2(
        basis: List[RydStateFS],
        H0: np.ndarray,
        Eac: float,
        ellipticity: float,
        field_omega: float,
        Edc: float,
        theta: float,
        n_max: int,
        varied: Tuple[str, np.ndarray] = None,
        energy_bands: bool = False,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    produces, then diagonalizes a floquet hamiltonian at a variety of different system parameters.

    independent variable is defined in parameter varied, it overwrites the value of the steady state

    eigenvectors and eigenvalues are organized into a large ndarray that allows the user to track
    eigenvalues as bands in the resulting floquet spectrum as the independent variable is varied.
    This is done by having the eigenvalues at each value of the independent variable be lined up
    to maximize the overlap of each indexed eigenvector with the eigenvector for the previous
    independent variable value.

    Args:
        basis : List of undressed atomic states to consider when creating the floquet hamiltonian
        H0 : Hamiltonian describing unperturbed atomic system. Expected to be diagonal, indexing
            should be the same as that of basis. Matrix elements should be reported in radial
            frequency (radians/s)
        Eac : Peak Electric field strength of the dressing field at the atoms (V/m)
        ellipticity : ellipticity of the dressing field at the atoms. Field polarization is
            described as: e_ac = sqrt(1-ellipticity)e_pi + sqrt(ellipticity)e_+
        field_omega : oscillation frequency of the dressing field in radial frequency. (radians/s)
        Edc : Electric field strength of the DC field at the atoms (V/m)
        theta : angle between the DC electric field and the quantization axis (radians)
        n_max : maximum number of fourier components of the dressing field to consider in the
            creation of the Floquet hamiltonian
        varied : (name of independent variable, values taken by independent variable), name must
            be one of the following ["Edc", "Eac", "ellipticity", "theta"]. values must be 1D array
            of floats. This likely will only work if values start from a small (preferably 0) value
            and make their way up to the maximal value.
        energy_bands : If true, sort bands within a fine structure level (eg |52P3/2;-1>) by energy,
            otherwise bands are sorted by m_j
    Returns:
        energies, eigenvectors: ndarrays containing the values of the system's energies and
            eigenvectors at each value of varied[1]
            energies.shape = ( (2*n_max+1)len(basis), len(varied[1] )
            eigenvectors.shape = ( (2*n_max+1)len(basis), (2*n_max+1)len(basis), len(varied[1] )
    """
    if type(varied[0]) is not str:
        raise TypeError(
            "first entry in varied parameter must be a string indicating which system parameter "
            "is an independent variable")

    flen = len(basis)*(2*n_max+1)
    print(flen)
    energies = np.ones((flen, len(varied[1])), dtype=float)
    eigenstates = np.zeros((flen, flen, len(varied[1])), dtype=complex)

    levels = implied_levels(basis)
    print("Levels :\n")
    basis_print(levels)
    starts = level_starts(levels)
    ips_summer = level_projector(basis, levels, n_max)

    dipoles_z = dipoles(basis, qs=[1, 0, 0])
    if varied[0] != "ellipticity":
        # dipoles_ac = dipoles(basis, qs=[np.sqrt(1-ellipticity), np.sqrt(ellipticity), 0])
        dipoles_ac = dipoles(basis, qs=[
            np.sqrt(1-ellipticity),
            -np.sqrt(ellipticity)/np.sqrt(2),
            -np.sqrt(ellipticity)/np.sqrt(2)
        ])
    else:
        dipoles_ac = None
    for i, ival in enumerate(varied[1]):
        if varied[0] == "Edc":
            Edc = ival
        elif varied[0] == "Eac":
            Eac = ival
        elif varied[0] == "ellipticity":
            ellipticity = ival
        elif varied[0] == "theta":
            theta = ival
        else:
            raise ValueError(f"independent variable {varied[0]} not recognized")

        t_start = time()
        eigenvalues, eigenvectors = floquet_diag(
            basis,
            H0,
            Eac,
            ellipticity,
            field_omega,
            Edc,
            theta,
            n_max,
            **{"dipoles_z": dipoles_z, "dipoles_ac": dipoles_ac}
        )

        print(f"floquet_diag call completed in {time()-t_start}s")
        print(f"Diagonalization complete for independent variable entry, value {i}, {ival}")

        # re-arrange eigenvalues and eigenvectors
        t_start = time()
        # first entry is special
        if i == 0:
            # compute overlaps wrt unperturbed eigenstates
            ips = np.abs(eigenvectors)**2
        else:
            ips = np.abs(np.dot(np.conj(eigenvectors), eigenstates[..., i - 1].T)) ** 2

        inds = np.argwhere(ips > 0.5)
        check0 = all([a in inds[:, 0] for a in range(flen)])
        check1 = all([a in inds[:, 1] for a in range(flen)])
        ts = time()
        if not (check0 and check1):
            missinds = [a for a in range(flen) if a not in inds[:, 1]]
            for j in missinds:
                kinds = [np.argsort(-ips[:, j])][0]
                # print(ips[:,j][kinds])
                # print(kinds[0])
                count = 0
                while True:
                    k = kinds[count]
                    if k in inds[:, 0]:
                        count += 1
                    else:
                        # print(inds)
                        inds = np.append(inds, [[k, j]], axis=0)
                        # print(inds)
                        break
        te = time()
        print(f"re-arrange time = {te - ts}")
        eigenstates[inds[:, 1], :, i] = eigenvectors[inds[:, 0], :]
        energies[inds[:, 1], i] = eigenvalues[inds[:, 0]]
        check0 = all([a in inds[:, 0] for a in range(flen)])
        check1 = all([a in inds[:, 1] for a in range(flen)])
        if not (check0 and check1):
            raise RuntimeError("Oopsy woopsy")

    return energies, eigenstates


def check_polarizability(
        basis: List[RydStateFS],
        t_level: RydStateFS,
        H0: np.ndarray,
        dc_end: float,
        samples: int,
        comp: Dict[str, int],
        fields: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determines the polarizability of all states in basis for AC/DC field parameters
    provided in fields dict.

    Args:
        basis: list of atomic states considered in computation
        t_level: target atomic level around which the basis was constructed
        H0: Hamiltonian of unperturbed atom system (expected to be diagonal). Matrix
            elements should be reported in radial frequency (radians/s)
        dc_end: maximum electric field strength to be sampled. dc electric field values
            are sampled from 0 to dc_end
        samples: number of dc electric field strength values to take. System is prone to crashing when
            too few samples are provided. Stable for when 100 samples per 20 volt span
        fields: dict of AC and DC field values. Currently the following parameters are expected
            to be represented in fields:
                "Eac" : Electric field strength of the AC field (V/m)
                "ellipticity" : ellipticity of the AC field. Field polarization defined as
                    e_ac = sqrt(1-ellipticity)e_pi + sqrt(ellipticity)e_+
                "field_omega" : oscillation frequency of the AC. Reported in radial frequency (radians/s)
                "Edc" : Ignored if included in fields dict
                "theta" : angle between quantization axis and electric field direction (radians)
        comp: dict of computational parameters used for this computation. Keys:
            "max_det" : float, maximum energy difference between t_level and any level included
                in the basis used in the computation
            "dl" : int, maximum difference between t_level.l and the orbital angular momentum
                quantum number of other states in the basis, |l-lp| <= dl
            "n_max" : int, maximum number of fourier components of the AC field to include in the
                computation
    """
    n_max = comp["n_max"]

    dcs = ("Edc", np.linspace(0, dc_end, samples))
    energies, eigenstates = floquet_loop(basis, H0, n_max=n_max, varied=dcs, **fields)

    quad = lambda v, v0, a, e0: a * (v - v0) ** 2 + e0

    dim = len(basis) * (2 * n_max + 1)
    alphas = np.zeros(dim, dtype=float)
    dalphas = np.zeros(dim, dtype=float)
    for i in range(dim):
        band_energies = energies[i, :]
        guess = [
            0.0, (band_energies[-1] - band_energies[samples // 2]) / dc_end ** 2,
            band_energies[samples // 2]
        ]
        try:
            popt, pcov = curve_fit(quad, dcs[1], band_energies, p0=guess)
            perr = np.sqrt(np.diag(pcov))

            alphas[i] = popt[1]
            dalphas[i] = perr[1]
        except RuntimeError:
            alphas[i] = np.NaN
            dalphas[i] = np.NaN
    fields["Edc"] = dcs[1]
    eigen_save(t_level, comp, fields, energies, eigenstates)
    del energies
    del eigenstates
    return alphas, dalphas

#  -------------------------------------------------------------------------------------------------
#  == functions to save and load data ==
#  -------------------------------------------------------------------------------------------------


def eigen_save(
        t_level: RydStateFS,
        computational_parameters: Dict[str, float],
        field_parameters: Dict[str, Union[float, np.ndarray]],
        energies: np.ndarray,
        eigenstates: np.ndarray
):
    """
    Saves energy and eigenstate ndarrays to a .npy file, formatting the file to provide relevant
    info on computational and field parameters used
    Args:
        t_level : target rydberg level around which computations are centered
        computational_parameters : dict of relevant computational parameters. Keys must be:
            "max_det" : float, maximum energy difference between t_level and any level included
                in the basis used in the computation
            "dl" : int, maximum difference between t_level.l and the orbital angular momentum
                quantum number of other states in the basis, |l-lp| <= dl
            "n_max" : int, maximum number of fourier components of the AC field to include in the
                computation
        field_parameters : dict of AC and DC field parameters used in teh computation. Exactly
            one of the field parameter values must be an ndarray, this is taken to be the
            independent variable. Keys must be:
                "Eac" : float or ndarray: electric field strength(s) of the oscillating field (V/m)
                "ellipticity" : float or ndarray: value(s) of the ellipticity of the field
                "field_omega" : float or ndarray: oscillation frequency(ies) of the AC field
                    (radians/s)
                "Edc" : float or ndarray: electric field strength(s) of the DC field (V/m)
                "theta" : float or ndarray: value(s) for the angle between the DC field and the
                    quantization axis
        energies : eigenvalues associated with the Shirley-Floquet Hamiltonians produced by the
            above parameters
        eigenstates : eigenvectors associated with the Shirley-Floquet Hamiltonians produced by the
            above parameters
    """
    folder = os.path.join(RESULTS_FOLDER, f"t_level_{t_level.n}_{t_level.l}_{int(t_level.j*2)}-2")
    ivar = [key for key, value in field_parameters.items() if type(value) is np.ndarray][0]
    ivals = (min(field_parameters[ivar]), max(field_parameters[ivar]))
    its = len(field_parameters[ivar])

    ivar_s = FORMATTED[ivar].split("-")[0]
    if ivar in ["Eac", "Edc", "field_omega"]:
        f_name = f"scan_{ivar_s}-{ivals[0]:.0f}-{ivals[1]:.0f}-{its}_"
    elif ivar in ["ellipticity", "theta"]:
        f_name = f"scan_{ivar_s}-{ivals[0]:.3f}-{ivals[1]:.3f}-{its}_"
    else:
        f_name = "This is not a case we accounted for"
    f_name += "_".join(
        [FORMATTED[key].format(val) for key, val in field_parameters.items() if key != ivar]
    )
    sub_folder = f"max_det-{int(computational_parameters['max_det'])}."
    sub_folder += f"n_max-{computational_parameters['n_max']}."
    sub_folder += f"dl-{computational_parameters['dl']}"
    file_path = os.path.join(folder, sub_folder)
    if not os.path.isdir(RESULTS_FOLDER):
        os.mkdir(RESULTS_FOLDER)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    f_name_e = f_name + "_energies"
    f_name_s = f_name + "_eigenstates"
    np.save(os.path.join(file_path, f_name_e), energies)
    np.save(os.path.join(file_path, f_name_s), eigenstates)


def get_t_level(filepath: str) -> RydStateFS:
    """
    Extracts the t_level defined in the filepath given
    Args:
        filepath : path to a set of energies + eigenstates files

    Returns:
        the relevant target level for those files
    """
    tl = "t_level"
    t_string = None
    for folder in filepath.split("\\"):
        if folder[:len(tl)] == tl:
            t_string = folder[len(tl)+1:]
    if t_string is None:
        raise ValueError(f"filepath '{filepath}' does not contain a folder with eigen-system data")
    # print(f"t_string = {t_string}")
    qns = {}
    for i, qn in enumerate(t_string.split("_")):
        # print(f"qn = {qn}")
        if i == 0:
            qns["n"] = int(qn)
        elif i == 1:
            qns["l"] = int(qn)
        elif i == 2:
            qns["j"] = int(qn.split("-")[0])/2
    return RydStateFS(**qns)


def get_comp_settings(filepath) -> Dict[str, int]:
    """
    Loads computational settings used to create eigen-system data for file(s) in filepath
    Args:
        filepath : path to a file or directory containing pre-computed eigen-system data for a
            floquet analysis

    Returns:
        computational_parameters : dictionary with information on how eigensystem data was computed:
            "max_det": int, maximum allowed energy difference between the target level and levels
                included in the basis. Reported in radial frequency (radians/s)
            "dl" : int, maximum difference between t_level.l and the orbital angular momentum
                quantum number of other states in the basis, |l-lp| <= dl
            "n_max" : int, maximum number of fourier components of the AC field to include in the
                computation
    """
    t_s = "max_det"
    c_string = None
    for folder in filepath.split("\\"):
        if folder[:len(t_s)] == t_s:
            c_string = folder
    if c_string is None:
        raise ValueError(f"filepath {filepath} did not contain computational settings info")
    computational_parameters = {}
    for setting in c_string.split("."):
        key, val = setting.split("-")
        computational_parameters[key] = int(val)

    if set(computational_parameters.keys()) != {"max_det", "dl", "n_max"}:
        raise RuntimeError("Computational Parameters not correct")
    return computational_parameters


def get_field_params(filepath: str) -> Dict[str, Union[float, np.ndarray]]:
    """
    Extracts the field parameters to compute previous eigen-system info rom the path to a file
    containing said eigensystem info.
    Args:
        filepath : path to a file containig eigen-system information (eigenvalues or
            eigenvectors) for the diagonalization of a Shirley-Floquet system
    Returns:
        field_parameters: dict containing info describing fields that perturbed the atom. Keys:
            "Eac" : float or ndarray: electric field strength(s) of the oscillating field (V/m)
            "ellipticity" : float or ndarray: value(s) of the ellipticity of the field
            "field_omega" : float or ndarray: oscillation frequency(ies) of the AC field
                (radians/s)
            "Edc" : float or ndarray: electric field strength(s) of the DC field (V/m)
            "theta" : float or ndarray: value(s) for the angle between the DC field and the
                quantization axis
    """
    t_s = "scan"
    file = filepath.split("\\")[-1]
    if file[:len(t_s)] != t_s:
        raise ValueError(f"Filepath {filepath} does not point to a valid eigen-system file")

    field_parameters = {}
    for i, par in enumerate(file.split("_")[1:]):
        if i == 0:
            ivar_vals = par.split("-")
            ivar = ivar_vals[0]
            if ivar == "eps":
                # In the case of ellipticity it's more convenient to increment by sqrt of the value
                # Here we instill that as the default behavior
                ivals = np.linspace(
                    np.sqrt(float(ivar_vals[1])),
                    np.sqrt(float(ivar_vals[2])),
                    int(ivar_vals[3])
                )**2
            else:
                ivals = np.linspace(float(ivar_vals[1]), float(ivar_vals[2]), int(ivar_vals[3]))
            field_parameters[LONGER[ivar]] = ivals
        else:
            try:
                pars = par.split("-")
                full_parameter = LONGER[pars[0]]
                field_parameters[full_parameter] = float(pars[1])
            except KeyError:
                if par in ["eigenstates.npy", "energies.npy"]:
                    continue
                else:
                    raise
    return field_parameters


def eigen_load(filepath) -> Tuple[RydStateFS, Dict, Dict]:
    """
    Loads the settings info for an eigen-system file from it's filepath
    Args:
        filepath : path to an .npy file containing energy or eigenstate info for a Shirley-Floquet
            computation

    Returns:
        t_level: Level around which the computation was performed
        computational_parameters: dict describing the computational parameters used to compute
            the data in the file
        field_parameters: dict describing the field parameters used to compute the data in the file
    """
    t_level = get_t_level(filepath)
    computational_parameters = get_comp_settings(filepath)
    field_parameters = get_field_params(filepath)

    return t_level, computational_parameters, field_parameters


def eigen_list(
    t_level: RydStateFS,
    computational_parameters: Dict[str, float]
) -> List[Dict[str, float]]:
    """
    Lists all files that were created by using the provided t_level and computational parameters
    Args:
        t_level : "Target Level" level around which the basis is centered
        computational_parameters :  dict of relevant computational parameters. Keys must be:
            "max_det" : float, maximum energy difference between t_level and any level included
                in the basis used in the computation
            "dl" : int, maximum difference between t_level.l and the orbital angular momentum
                quantum number of other states in the basis, |l-lp| <= dl
            "n_max" : int, maximum number of fourier components of the AC field to include in the
                computation
    Returns:
        field_parameter_list:
            list of dicts. Each corresponding to a pair of files, listing the field_parameters
            used to compute the contents of said files. Keys:
            "f_name": file name of the files
            "Eac" : float or ndarray: electric field strength(s) of the oscillating field (V/m)
            "ellipticity" : float or ndarray: value(s) of the ellipticity of the field
            "field_omega" : float or ndarray: oscillation frequency(ies) of the AC field
                (radians/s)
            "Edc" : float or ndarray: electric field strength(s) of the DC field (V/m)
            "theta" : float or ndarray: value(s) for the angle between the DC field and the
                quantization axis
    """

    # the directory naming function rounds comp parameters to ints. We do that here to compensate
    computational_parameters = {key: int(value) for key, value in computational_parameters.items()}
    comp_path = None
    for level_directory in os.listdir(RESULTS_FOLDER):
        try:
            dir_level = get_t_level(level_directory)
        except ValueError:
            # ValueErrors are raised when the directory is not relevant
            continue
        if [dir_level.n, dir_level.l, dir_level.j] == [t_level.n, t_level.l, t_level.j]:
            level_path = os.path.join(RESULTS_FOLDER, level_directory)
            for comp_dir in os.listdir(level_path):
                try:
                    dir_comp_params = get_comp_settings(comp_dir)
                except ValueError:
                    continue
                if dir_comp_params == computational_parameters:
                    comp_path = os.path.join(level_path, comp_dir)
                    # There should only be one of these folders.
                    break
            break
    if comp_path is None:
        raise FileNotFoundError(
            "No directories with the specified computational parameters were found"
        )
    field_param_list = [None]*(len(os.listdir(comp_path))//2)
    j = 0
    eigenstates = "eigenstates.npy"
    for eigen_file in os.listdir(comp_path):
        # each scan has two files associated with it. Choose eigenstates arbitrarily to prevent
        # double counting
        if eigen_file[-len(eigenstates):] == eigenstates:
            field_params = get_field_params(eigen_file)
            eg_name = eigen_file[:-len(eigenstates)] + "{}.npy"  # so both files can be
            # references w/ .format calls
            field_params.update({"filename": os.path.join(comp_path, eg_name)})
            field_param_list[j] = field_params
            j += 1
        else:
            continue

    return field_param_list


def eigen_find(
        t_level: RydStateFS,
        computational_parameters: Dict[str, int],
        field_parameters: Dict[str, float]
) -> List[Tuple[str, float, float]]:
    """
    If there is a file within the filesystem that contains info from the scan prescribed by the
    args, this function finds it and returns the limits of the given file relevant to the scan
    described
    Args:
        t_level : "Target Level" level around which the basis is centered
        computational_parameters :  dict of relevant computational parameters. Keys must be:
            "max_det" : float, maximum energy difference between t_level and any level included
                in the basis used in the computation
            "dl" : int, maximum difference between t_level.l and the orbital angular momentum
                quantum number of other states in the basis, |l-lp| <= dl
            "n_max" : int, maximum number of fourier components of the AC field to include in the
                computation
        field_parameters : dict of AC and DC field parameters used in teh computation. Exactly
            one of the field parameter values must be an ndarray, this is taken to be the
            independent variable. Keys must be:
                "Eac" : float or ndarray: electric field strength(s) of the oscillating field (V/m)
                "ellipticity" : float or ndarray: value(s) of the ellipticity of the field
                "field_omega" : float or ndarray: oscillation frequency(ies) of the AC field
                    (radians/s)
                "Edc" : float or ndarray: electric field strength(s) of the DC field (V/m)
                "theta" : float or ndarray: value(s) for the angle between the DC field and the
                    quantization axis
    Returns:
        list of the following info per-item
        filename, lower_bound, upper_bound:
            if there is a file containing a part of the scan this function will return the (
            partial) filename, the lower limit of the relevant section of that file,
            and the upper limit of the relevant section of that file.
            ie. If the specified scan varies Edc from -3 to 13, and a file has all other
            variables the same, but varies Edc from 0 to 20, 0 and 13 are returned as the lower
            and upper bounds respectively.
    """
    comp_files = eigen_list(t_level, computational_parameters)
    # find the independent variable in the scan
    for param, value in field_parameters.items():
        try:
            tmp = value[1]
        except TypeError:
            continue
        else:
            varied = (param, value)
            break

    good_scans: List[Tuple[str, int, int]] = []
    for scan in comp_files:
        min_scan = {
            key: value for key, value in scan.items() if key not in ["filename", varied[0]]}
        min_param = {}
        for key, value in field_parameters.items():
            if key == varied[0]:
                continue
            if key in ["Eac", "Edc", "field_omega"]:
                min_param[key] = round(value, 0)
            elif key in ["ellipticity", "theta"]:
                min_param[key] = round(value, 3)
#        print(min_scan)
#        print(min_param)
        try:
            if min_scan == min_param:
                s_ivals = scan[varied[0]]
                low = max(min(s_ivals), min(varied[1]))
                high = min(max(s_ivals), max(varied[1]))
                if high-low > 0:
                    good_scans.append((scan["filename"], low, high))
        except ValueError as err:
            # min_scan == min_param breaks if scan is of different independent variable. We just
            # handle that here
            # print(err)
            continue
    return good_scans


def eigen_terpolate(
        fname: str,
        low: float,
        high: float,
        field_parameters: Dict[str, Union[float, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads eigensystem data from the file provided, then interpolates the data to match up with
    the sampling of field parameters (low to high) and returns a partially complete
    section of the energies and eigenstates ndarrays that describe the bands of the floquet system.
    
    Raises:
        RuntimeError: Runtime error is raised if sampling rate in field_params is significantly 
            higher than that of the provided file. Can also be raised if interpolation fails for 
            any reason.
    
    Args:
        fname : partial filename of the eigensystem files to be loaded
        low : lower bound of shared values between the stored data and the desired data
        high : upper bound of shared values between the stored data and the desired data
        field_parameters : dict of AC and DC field parameters used in teh computation. Exactly
            one of the field parameter values must be an ndarray, this is taken to be the
            independent variable. Keys must be:
                "Eac" : float or ndarray: electric field strength(s) of the oscillating field (V/m)
                "ellipticity" : float or ndarray: value(s) of the ellipticity of the field
                "field_omega" : float or ndarray: oscillation frequency(ies) of the AC field
                    (radians/s)
                "Edc" : float or ndarray: electric field strength(s) of the DC field (V/m)
                "theta" : float or ndarray: value(s) for the angle between the DC field and the
                    quantization axis

    Returns:
        energies, eigenvalues: partial solutions to the desired eigensystem based on loaded +
            interpolated data from the provided file.
    """
    f_level, f_comp, f_fields = eigen_load(fname.format("energies"))

    for param, value in f_fields.items():
        try:
            tmp = value[1]
        except TypeError:
            f_ivar = param
            f_ivals = value

    for param, value in field_parameters.items():
        try:
            tmp = value[1]
        except TypeError:
            ivar = param
            ivals = value

    # check files sampling rate and compare to desired sampling rate
    f_rate = (max(f_ivals) - min(f_ivals)) / len(f_ivals)
    s_rate = (max(ivals) - min(ivals)) / len(ivals)
    if s_rate > f_rate * 2:
        raise RuntimeError("File sample rate much lower than desired sample rate.")

    f_energies = np.load(fname.format("energies"))
    f_eigenstates = np.load(fname.format("eigenstates"))

    # Make interpolating functions for f_energies and f_eigenstates
    energy_interpolator = interpolate(f_ivals, f_energies)
    state_interpolator = interpolate(f_ivals, f_eigenstates)

    energies = np.zeros((f_energies.shape[0], len(ivals)), dtype=float)
    eigenstates = np.zeros((f_energies.shape[0], f_energies.shape[0], len(ivals)), dtype=complex)
    # sample f_energies and f_eigenstates for each overlapped value in ivals
    for i, val in enumerate(ivals):
        if low <= i <= high:
            energies[:, i] = energy_interpolator(val)
            eigenstates[..., i] = state_interpolator(val)

    return energies, eigenstates


def floquet_loop_loaded(
        t_level,
        basis,
        computational_parameters,
        field_parameters
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scans through independent variable specified in field_parameters. For each value of the
    independent variable, generates then diagonalizes a Shirley-Floquet Hamiltonian that
    describes the system as specified by computational_parameters and field_parameters.
    Afterwords it re-arranges the eigenvalues and eigenvectors that are the result of the
    diagonalization so that bands are tracked as they evolve with the independent variable.

    Makes use of the filesystem designed by this package to first check if all or part of the scan
    desired is already provided in any of the files produced by previous scans. If so it loads
    the relevant section(s) of that(those) file(s), interpolating the loaded data where necessary
    and appropriate, to evaluate the system at the relevant points.


    Example Use:
    t_level = RydStateFS(52,1,3/2)
    comp = {"n_max" = 3, "dl" = 1, "max_det" = 50e9*2*pi}
    basis = build_basis(t_level, **comp)
    fields = {
        "field_omega" : 4780e9*2*pi,
        "Eac" : np.linspace(0,80,200),  # scan from 0V/m to 80V/m in 80 steps
        "ellipticity" : 0.027,  # 2.7% ellipticity in AC field
        "Edc" : 0,  # V/m
        "theta" : 0.2*pi  # angle between quantization axis and DC field, in radians
    }
    energies, eigenstates = floquet_loop_loaded(t_level, comp, fields)

    Args:
        t_level : target rydberg level around which computations are centered
        basis :
        computational_parameters : dict of relevant computational parameters. Keys must be:
            "max_det" : float, maximum energy difference between t_level and any level included
                in the basis used in the computation
            "dl" : int, maximum difference between t_level.l and the orbital angular momentum
                quantum number of other states in the basis, |l-lp| <= dl
            "n_max" : int, maximum number of fourier components of the AC field to include in the
                computation
        field_parameters : dict of AC and DC field parameters used in teh computation. Exactly
            one of the field parameter values must be an ndarray, this is taken to be the
            independent variable. Keys must be:
                "Eac" : float or ndarray: electric field strength(s) of the oscillating field (V/m)
                "ellipticity" : float or ndarray: value(s) of the ellipticity of the field
                "field_omega" : float or ndarray: oscillation frequency(ies) of the AC field
                    (radians/s)
                "Edc" : float or ndarray: electric field strength(s) of the DC field (V/m)
                "theta" : float or ndarray: value(s) for the angle between the DC field and the
                    quantization axis
    Returns:
        energies, eigenvectors: ndarrays containing the values of the system's energies and
            eigenvectors at each value of the independent variable
            energies.shape = ( (2*n_max+1)len(basis), len(varied[1] )
            eigenvectors.shape = ( (2*n_max+1)len(basis), (2*n_max+1)len(basis), len(varied[1] )
    """
    n_max = computational_parameters["n_max"]
    ivar, ivals = None, None
    for param, value in field_parameters.items():
        try:
            tmp = value[1]
        except TypeError:
            continue
        else:
            ivar = param
            ivals = value

    if ivar is None:
        raise ValueError(
            "No independent variable provided. Exactly one value in field dict must be iterable"
        )

    flen = len(basis) * (2 * n_max + 1)
    old_scans = eigen_find(t_level, computational_parameters, field_parameters)
    if len(old_scans) == 0:
        low, high = None, None
        energies = np.zeros((flen, len(ivals)), dtype=float)
        eigenstates = np.zeros((flen, flen, len(ivals)), dtype=complex)
    elif len(old_scans) >= 1:  # TODO: implement support for multiple relevant scans if necessary
        fname, low, high = old_scans[0]
        try:
            energies, eigenstates = eigen_terpolate(fname, low, high, field_parameters)
        except RuntimeError:
            energies = np.zeros((flen, len(ivals)), dtype=float)
            eigenstates = np.zeros((flen, flen, len(ivals)), dtype=complex)

    H0 = build_H0(basis, t_state=t_level, offset_zeeman=int(1e3), j_split=0)

    fields = {key: value for key, value in field_parameters.items()}  # Hacky deep-copy

    if ivar != "ellipticity":
        qs = [
            np.sqrt(1-fields["ellipticity"]),
            np.sqrt(fields["ellipticity"]),
            0
        ]
        dipoles_ac = dipoles(basis, qs)
    else:
        dipoles_ac = 0
    dipoles_z = dipoles(basis, [1, 0, 0])

    f_kwargs = fields
    f_kwargs.update(computational_parameters)
    f_kwargs.update({"dipoles_z": dipoles_z, "dipoles_ac": dipoles_ac})

    flen = len(basis)*(2*n_max+1)
    energies = np.zeros((flen, len(ivals)), dtype=float)
    eigenstates = np.zeros((flen, flen, len(ivals)), dtype=complex)

    levels = implied_levels(basis)
    print("Levels :\n")
    basis_print(levels)
    starts = level_starts(levels)
    ips_summer = level_projector(basis, levels, n_max)

    for i, val in enumerate(ivals):
        f_kwargs[ivar] = val

        t_start = time()
        eigenvalues, eigenvectors = floquet_diag(basis, H0, **f_kwargs)

        print(f"floquet_diag call completed in {time()-t_start}s")
        print(f"Diagonalization complete for independent variable entry, value {i}, {val}")

        # re-arrange eigenvalues and eigenvectors
        t_start = time()
        # first entry is special
        if i == 0:
            # compute overlaps wrt unperturbed eigenstates
            ips = np.abs(eigenvectors)**2
            # sum over all zeeman states in each level
            ips_levels = np.dot(ips, ips_summer)
            # sum over all level in each fourier sub basis
            used_inds = []
            for j, level in enumerate(levels):
                for k, n in enumerate(range(-n_max, n_max+1)):
                    print(f"finding good eigenvectors for |level, n> = |{level.ket()},{n}>")
                    # eigenvectors that have >50% population in this level
                    thrsh = 0.5
                    inds_l = np.argwhere(ips_levels[:, j+k*len(levels)] > thrsh)
                    # if the 50% threshold is too high to accommodate all m levels, lower the
                    # threshold incrementally
                    while len(inds_l) < 2*level.j+1:
                        thrsh *= 0.95
                        print(f"expansion required, threshold reduced to {thrsh}")
                        inds_l = np.argwhere(ips_levels[:, j + k * len(levels)] > thrsh)

                    # find the band that has the greatest overlap with each mj level
                    strt = starts[j]+k*len(basis)
                    for a in range(int(2*level.j + 1)):
                        # print(f"m = {-level.j + a}")
                        # print(f"sub_ips = {ips[inds_l, strt + a]}")
                        ev_ind = np.argmax(ips[inds_l, strt + a])
                        if inds_l[ev_ind, 0] in used_inds:
                            print(f"WARNING: index {inds_l[ev_ind, 0]} has been used")
                        used_inds.append(inds_l[ev_ind, 0])
                        # print(f"new inds = {inds_l[ev_ind,0]}")
                        # print(f"New energy = {eigenvalues[inds_l[ev_ind, 0]]}")
                        eigenstates[strt + a, :, 0] = eigenvectors[inds_l[ev_ind, 0], :]
                        energies[strt + a, 0] = np.real(eigenvalues[inds_l[ev_ind, 0]])
        else:
            # try doing this the easy way. Find all overlaps of the computed eigenstates with the
            # previous iteration's eigenstates. For each eigenstate find the previous eigenstate
            # that has >50% overlap. Use those indeces as a map for current eigenstates to previous
            # eigenstates
            ips = np.abs(np.dot(np.conj(eigenvectors), eigenstates[..., i-1].T))**2
            inds = np.argwhere(ips > 0.5)

            # check that all basis states are represented in each column of inds:
            check0 = all([a in inds[:, 0] for a in range(flen)])
            check1 = all([a in inds[:, 1] for a in range(flen)])
            if check0 and check1:
                for ind in inds:
                    eigenstates[ind[1], :, i] = eigenvectors[ind[0], :]
                    energies[ind[1], i] = eigenvalues[ind[0]]
            else:
                # if the easy way fails, we find the eigenvectors that maximize overlap with
                # previous eigenvectors by first finding which levels have >50% population, then
                # finding which band within that level has the greatest overlap with the
                # eigenvectors. We then check that all states in the basis are represented by the
                # resulting index list
                print(inds)
                print(ips.shape, ips)
                ips_levels = np.dot(ips, ips_summer)
                inds_l = np.argwhere(ips_levels > 0.5)
                if all([a in inds_l[:, 0] for a in range(flen)]):
                    for ind_l in inds_l:
                        lvl = ind_l[1]%len(levels)
                        n_ind = int(ind_l[1]/len(levels))
                        strt = starts[lvl] + n_ind * len(basis)
                        stp = starts[lvl + 1] + n_ind * len(basis)
                        sub_ips = ips[ind_l[0], strt: stp]
                        sub_ips = sub_ips/sub_ips.sum()  # normalize
                        act_ind = np.argwhere(sub_ips == max(sub_ips))
                        ind = [ind_l[0], strt + act_ind]
                        eigenstates[ind[1], :, i] = eigenvectors[ind[0], :]
                        energies[ind[1], i] = eigenvalues[ind[0]]
                else:
                    # return for debugging
                    # return energies, eigenstates, ips, inds, ips_levels, inds_l
                    raise RuntimeError(
                      f"Failed to represent all basis states\n{inds_l}")
        print(f"re-arrangement done after time {time()-t_start}s")

    return energies, eigenstates