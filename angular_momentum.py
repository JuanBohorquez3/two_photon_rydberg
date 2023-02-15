# Useful functions and constants having to do with angular momentum
import numpy as np
from typing import TypeVar
from scipy.special import factorial

from basics import *

moment = TypeVar("moment", int, float)


def d_jmm(beta: float, j: moment, m: moment, mp: moment) -> float:
    """
    Computes matrix element of little d rotation operator for given j, m, mj values

    Matrix constructed by this operator creates rotations about y by angle beta
    Args:
        beta : rotation angle (radians)
        j : total angular momentum quantum number. Integer of half-integer
        m : azimuthal angular momentum quantum number of un-primed state, Integer or half-integer
        mp : azimuthal angular momentum quantum number of primed state, Integer or half-integer

    Returns:
        d_jmm : matrix element of the little d rotation operator
    """
    k_min = int(max(0, -m-mp))
    k_max = int(min(j-m, j-mp))
    # print(f"j,m,mp = {j,m,mp}")
    # print(f"\tk_min={k_min} k_max={k_max}")
    sgn = (-1)**(j-mp)
    roots = np.sqrt(
        factorial(j+m)*factorial(j-m)*factorial(j+mp)*factorial(j-mp)
    )
    # print(f"\t roots = {roots}")

    sum = 0
    for k in range(k_min,k_max+1):
        # print(f"\t\tk = {k}")
        denom = factorial(k)*factorial(j-m-k)*factorial(j-mp-k)*factorial(m+mp+k)
        # print(f"\t\tdenom = {denom}")
        sum += (-1)**k * (np.cos(beta/2)**(m+mp+2*k) * np.sin(beta/2)**(2*j-m-mp-2*k))/denom
    return sgn*roots*sum