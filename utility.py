import numpy as np
from arc import *
from typing import List, Tuple
pi = np.pi

from basics import *


class RydStateFS:
    """
    Tiny wrapper class to hold quantum numbers of rydberg state in FS basis
    """
    def __init__(self, n, l, j, **kwargs):
        """
        Args:
            n : principle quantum number (int)
            l : orbital angular momentum quantum number (int)
            j : FS angular momentum quantum number (int/half int)
        kwargs:
            for additional desired quantum numbers
            atom (str): default = "Cesium". Species of atom
        """
        self.n = n
        self.l = l
        self.j = j
        self.__qn = {"n": n, "l": l, "j": j}
        self.__qn.update(kwargs)
        if "atom" in kwargs.keys():
            self.atom_name = kwargs["atom"]
        else:
            self.atom_name = "Cesium"
        self.atom = eval(f"{self.atom_name}()")

    def __getitem__(self, arg):
        return self.__qn[arg]

    def quantum_numbers(self):
        return self.__qn

    def get_energy(self):
        """Wraps arc atom.getEnergy function. Returns level's energy in eV (relative to
        ionization threshold)"""
        return self.atom.getEnergy(self.n, self.l, self.j)

    def __repr__(self):
        return f"RydStateFS({','.join([f'{key}={value}' for key,value in self.__qn.items()])})"

    def __str__(self):
        return self.__repr__()
    # Comparison operators check energy levels of compared states

    def __lt__(self, other):
        return self.get_energy() < other.get_energy()

    def __eq__(self, other):
        return self.get_energy() == other.get_energy()

    def ket(self, bare=False):
        L_labels = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H"}
        if list(self.__qn.keys()) == ["n", "l", "j"] or bare:
            return f"|{self.n},{L_labels[self.l]},{int(2*self.j)}/2>"
        else:
            extra_qn = {q_l: q_v for q_l, q_v in self.__qn.items() if q_l not in ["n","l","j"]}
            for q_l, q_v in extra_qn.items():
                if type(q_v) is not int:
                    extra_qn[q_l] = f"{int(q_v*2)}/2"
            return f"|{self.n},{L_labels[self.l]},{int(2*self.j)}/2;" + ",".join([
                f"{q_l}:{q_v}" for q_l, q_v in extra_qn.items()
            ]) + ">"

    def bra(self):
        L_labels = {0: "S", 1: "P", 2: "D",3:"F"}
        if list(self.__qn.keys()) == ["n", "l", "j"]:
            return f"<{self.n},{L_labels[self.l]},{int(2*self.j)}/2|"
        else:
            extra_qn = {q_l: q_v for q_l, q_v in self.__qn.items() if q_l not in ["n","l","j"]}
            for q_l, q_v in extra_qn.items():
                if type(q_v) is not int:
                    extra_qn[q_l] = f"{int(q_v*2)}/2"
            return f"<{self.n},{L_labels[self.l]},{int(2*self.j)}/2;" + ",".join([
                f"{q_l}:{q_v}" for q_l, q_v in extra_qn.items()
            ]) + "|"


def detuning(state1: RydStateFS, state2: RydStateFS, field_omega: float) -> float:
    """
    Returns detuning of field from bare resonance between state1 and state2
    Args:
        state1: Rydberg state in FS basis for the first state
        state2: Rydberg state in FS basis for the second state
        field_omega: radial frequency of a single tone field coupling state1 and state2 (Hz)
    Returns:
        detuning : field detuning from bare resonance in radial frequency (radian/s)
    """
    bare_res = (state1.get_energy()-state2.get_energy())*e/hb
    return bare_res - field_omega


def list_print(lst):
    return "[\n\t" + ",\n\t".join(f"{vl:.2f}" for vl in lst) + "]"


def matrix_print(matrix: np.ndarray, mult: float = 1):
    """
    Prints matrix in more legible format
    Args:
        matrix : matrix to be printed
        mult : value by which matrix elements are scaled
    """
    print(
        "\n".join([
            "[" + ",".join([f"{me.real*mult:2.1f}" for me in row]) + "]" for row in matrix
        ]))


def basis_print(basis: List[RydStateFS]):
    """
    Prints formatted basis
    Args:
        basis : List of Rydberg states to be printed
    """
    print("[\n\t" + ",\n\t".join([s.ket() for s in basis]) + "\n]")


def get_energy_nstate(state: RydStateFS, field_omega) -> float:
    """
    gets modified energy for a dressed photon-atom state. State must have a quantum number "nphot"
    Args:
        state : State whose energy is being calculated. Must have quantum number "nphot"
    Returns:
        state energy in radial frequency (radian/s)
    """

    return state.get_energy() * e / hb + state["nphot"] * field_omega


def build_basis_l(t_level: RydStateFS, max_det: float, dl: int = 1) -> List[RydStateFS]:
    """
    Identify and build out a basis of rydberg levels near the desired target state. Levels are
    selected to have abs(lp-l)<=1 with target state, and an energy difference
    (measured in radians/s) less than max_det.
    Args:
        t_level : target level. Level around which the basis is built
        max_det : maximum allowed transition frequency between t_level and any level within the
            basis
        dl : maximum allowed |l-l'| for states other than target state. Default is 1
            (only dipole allowed transitions)
    Returns:
        basis_l: list of all rydberg levels within 1 orbital angular momentum quantum number and the
            specified transition frequency range
    """
    def det(n, ll):
        return detuning(RydStateFS(n, ll, ll+1/2), t_level, 0)
    levels = []
    for lp in range(max(0, t_level.l-dl), t_level.l+dl+1):
        # print(f"Looking at lp = {lp}")
        if abs(det(t_level.n, lp)) < max_det:
            # print(f"states n, lp have good detunings {t_level.n},{lp}")
            min_n = t_level.n
            while abs(det(min_n-1, lp)) < max_det:
                min_n -= 1
            max_n = min_n
            while abs(det(max_n+1, lp)) < max_det:
                max_n += 1
            good_ns = range(min_n, max_n+1)
        else:
            # print(f"States n, lp have bad detunings {t_level.n}, {lp}")
            sgn = np.sign(det(t_level.n, lp))
            n_p = t_level.n
            good_ns = []
            # print("Looping through ns")
            while sgn*det(n_p, lp) >= -max_det:
                # print(f"n_p = {n_p}, D = {det(n_p,lp)*1e-9/(2*pi):.2f}")
                if abs(det(n_p, lp)) < max_det:
                    # print(f"energy was good. {det(n_p,lp)*1e-9/(2*pi):.2f} GHZ")
                    good_ns.append(n_p)
                n_p -= int(sgn)
        if lp > 0:
            levels.extend([RydStateFS(n, lp, lp-1/2) for n in good_ns])
        levels.extend([RydStateFS(n, lp, lp+1/2) for n in good_ns])
    return levels


def expand_basis_z(levels: List[RydStateFS], single_side=True) -> List[RydStateFS]:
    """
    expands basis defined by FS rydberg levels to include zeeman sub-levels.
    Only positive sub-levels are included currently
    Args:
        levels : Rydberg levels to consider in basis
        single_side : produce only mj levels >= 0?
    Returns:
        basis: basis or relevant rydberg states in the FS basis, with magnetic
            sub-levels. Basis is sorted by rydberg state energy
    """
    basis = []
    for level in levels:
        if level.j < 0:
            continue
        mjs = np.arange(-level.j * (not single_side) + 1/2 * single_side, level.j + 1)
        basis.extend([RydStateFS(level.n, level.l, level.j, mj=m) for m in
                      mjs])
    basis.sort()
    return basis


def build_basis(
        t_level: RydStateFS,
        max_det: float,
        single_side=True,
        dl: int = 1
) -> Tuple[List[RydStateFS], List[RydStateFS]]:
    """
    Identify and build out a basis of rydberg levels near the desired target state. Levels are
    selected to have abs(lp-l)<=1 with target state, and an energy difference
    (measured in radians/s) less than max_det. Basis is in FS basis with positive magnetic
    sub-levels
    Args:
        t_level : target level. Level around which the basis is built
        max_det : maximum allowed transition frequency between t_level and any level within the
            basis
        single_side : Produce only mj levels >= 0?
        dl : maximum allowed |l-l'| for states other than target state. Default is 1
            (only dipole allowed transitions)
    Returns:
        basis: list of all rydberg levels within 1 orbital angular momentum quantum number and the
            specified transition frequency range
    """
    levels = build_basis_l(t_level, max_det, dl)
    basis = expand_basis_z(levels, single_side)
    levels.sort()
    return levels, basis


def build_n_basis(basis: List[RydStateFS], n_max: int) -> List[RydStateFS]:
    """
    Expands a basis to include dressed n_photon states as in Shirley-Floquet formalism
    Args:
        basis : atomic basis states
        n_max : maximum number of photons to consider
    Returns:
        basis_n: list of dressed atom-photon number states
    """
    ns = range(-n_max, n_max+1)
    basis_n = []
    for n in ns:
        basis_n.extend([eval(state.__str__()[:-1]+f",nphot={n})") for state in basis])
    return basis_n


def implied_levels(basis: List[RydStateFS]) -> List[RydStateFS]:
    """
    Finds FS energy levels represented by basis
    Args:
        basis : atomic basis states

    Returns:
        levels : atomic energy levels in basis
    """
    levels = []
    for state in basis:
        if len(levels) == 0:
            levels.append(RydStateFS(state.n, state.l, state.j))
            continue
        if [state.n, state.l, state.j] == [levels[-1].n, levels[-1].l, levels[-1].j]:
            # print(f"state {state.ket()} == level {levels[-1].ket()}")

            continue
        else:
            levels.append(RydStateFS(state.n, state.l, state.j))
    return levels


def level_starts(levels: List[RydStateFS]) -> List[int]:
    """
    use list of atomic levels to determine which indices in a zeeman resolved basis correspond to
    the start of each level
    Args:
        levels : list of all FS rydberg energy levels make up basis

    Returns:
        starts : list of ints. Each int corresponds to the first index in the basis that represents
            the first entry from that level. Last entry is length of basis array, for utility
            reasons.
    """
    starts = [0] * (len(levels) + 1)
    for i, level in enumerate(levels):
        starts[i + 1] = starts[i] + int(2 * level.j + 1)

    return starts


def level_projector(basis, levels, n_max) -> np.ndarray:
    """
    matrix that when multiplied with a matrix of eigenvector overlaps (on the right) produces matrix
    of eigenvector probabilities with each energy level summed over.

    example usage:
        lvl_prj = level_projector(basis, levels, n_max)
        ips = np.absolute(np.dot(eigenvectors_old,eigenvectors_new.T))**2
        ips_levels = np.dot(ips, lvl_prj)

    A full example:
    If a basis is composed of states in energy levels |52S1/2> |52P3/2>, with n_max = 0
    If a hamiltonian for that basis is diagonalized, and the following eigenvectors are found:
    [ [1/sqrt(2),1/sqrt(2),0,0,0,0],
      [1/sqrt(2),-1/sqrt(2),0,0,0,0],
      [0,0,1/2,1/2,1/2,1/2],
      [0,0,1/2,1/2,-1/2,-1/2] ]

    The ips with the corresponding diagonal basis (ie the basis corresponding to the unperturbed
    atomic hamiltonian (given a tiny zeeman shift)) would be:
    [ [1/2,1/2,0,0,0,0],
      [1/2,1/2,0,0,0,0],
      [0,0,1/4,1/4,1/4,1/4],
      [0,0,1/4,1/4,1/4,1/4],
      [0,0,1/4,1/4,1/4,1/4],
      [0,0,1/4,1/4,1/4,1/4] ]

    corresponding to the probability of finding each eigenvector of the perturbed hamiltonian in a
    given basis state.

    the result of the line np.dot(ips,level_projector(basis, level, n_max)) would be
    [ [1,0],
      [1,0],
      [0,1],
      [0,1],
      [0,1],
      [0,1] ]


    Args:
        basis : list of atomic (undressed) states that considered in computation
        levels : list of energy levels that make up basis
        n_max : maximum number of fourier components being considered in Shirley-Floquet
            computations

    Returns:
        level_projector: matrix that sums populations over all states in each zeeman manifold.
            level_projector.shape = ( (2*n_max+1)*len(basis), (2*n_max+1)*len(levels) )
    """
    ns = 2 * n_max + 1
    proj = np.zeros((len(basis)*ns, len(levels * ns)), dtype=float)
    starts = level_starts(levels)
    for j, n in enumerate(range(-n_max,n_max+1)):
        for i, level in enumerate(levels):
            proj[starts[i]+j*len(basis):starts[i+1]+j*len(basis), i+j*len(levels)] = 1
    return proj

