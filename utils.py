import numpy as np
import sys
sys.path.append("./material")
from hw_utils import (
    polynomial_coeff_to_reflection_coeff,
    reflection_coeff_to_polynomial_coeff,
)

def decoding_coeff(LARc, A, B):
    LARd = np.zeros(8) # decoded LARc values
    rd = np.zeros(8) # decoded reflection coefficients
    for i in range(8):
        LARd[i] = (LARc[i] - B[i]) / A[i] # decoding of LARc values

        # Interpolation of the log-Area Rations (optional)

        # Transformation of the log-Area Rations into reflection coefficients
        if np.abs(LARd[i]) < 0.675:
            rd[i] = LARd[i]
        elif np.abs(LARd[i] >= 0.675 and np.abs(LARd[i] < 1.225)):
            rd[i] = np.sign(LARd[i]) * (0.5 * np.abs(LARd[i] + 0.3375))
        elif np.abs(LARd[i] >= 1.225 and np.abs(LARd[i]) <= 1.625):
            rd[i] = np.sign(LARd[i]) * (0.125 * np.abs(LARd[i] + 0.796875))

    a, e_final = reflection_coeff_to_polynomial_coeff(rd) # convert reflection coeffs to poly coeffs

    coeffs = [a[0], -a[1], -a[2], -a[3], -a[4], -a[5], -a[6], -a[7], -a[8]] # FIR filter coefficients

    return coeffs
