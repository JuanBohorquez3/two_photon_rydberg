import numpy as np
from typing import List


class SphericalVector:
    """
    Class to track the weights of polarization for an vector in cartesian space, describing it in
    a spherical basis.

    Represents a unit vector as a spherical tensor operator. If the weights provided don't
    describe a vector of length 1, the vector is normalized
    """
    def __init__(self, vector: List[complex], spherical_basis: bool = True):
        """
        Args:
            vector : vector to be described in the spherical basis. Should be a length three list of
                the different components of the vector.
                If spherical_basis is True, vector should be indexed [e_0, e_+1, e_-1]
                Otherwise, vector should be indexed [e_x,e_y,e_z]
            spherical_basis : If True, the vector input is already in the spherical basis. Otherwise
                the vector input is treated as if it's been input in the cartesian basis
        """

        try:
            tmp = vector[0]
        except IndexError:
            raise TypeError("Vector argument must be iterable, and indexable")
        if len(vector) != 3:
            raise ValueError("Spherical Vectors have exactly three components.")
        vector = np.array(vector) / self._mag(vector)

        self.__vector = vector if spherical_basis else self._to_sphere(vector)

    @staticmethod
    def _mag(vector: List[complex]) -> float:
        """
        computes the magnitude of a vector
        Args:
            vector : length three list describing a vector in either spherical or cartesian basis.

        Returns:
            magnitude of the input vector
        """
        return np.sqrt(sum([comp*comp.conjugate() for comp in vector]))

    @staticmethod
    def _to_sphere(vector: List[complex]) -> List[complex]:
        """
        Performs basis change for a vector in the cartesian basis into a vector in the spherical
        basis.
        Args:
            vector : length three list describing a vector in the cartesian basis.

        Returns:
            new_vector : the same vector as above in the spherical basis
        """
        [x, y, z] = [0, 1, 2]
        new_vec = [0, 0, 0]
        new_vec[0] = vector[z]
        new_vec[1] = -(vector[x] + 1j * vector[y]) / np.sqrt(2)
        new_vec[-1] = (vector[x] + 1j * vector[y]) / np.sqrt(2)

        return new_vec

    def __getitem__(self, item):
        return self.__vector[item]

    def __len__(self):
        return len(self.__vector)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__vector})"

    def __iter__(self):
        return self.__vector.__iter__()