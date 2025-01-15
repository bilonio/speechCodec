import numpy as np
import sys

sys.path.append("./material")
from hw_utils import reflection_coeff_to_polynomial_coeff
from scipy.signal import lfilter


def RPE_frame_st_decoder(LARc, curr_frame_st_resd):
    # Decoding of the quantized Log-Area ratios
    A = [20, 20, 20, 20, 13.637, 15.00, 8.334, 8.824]
    B = [0, 0, 4, -5, 0.184, -3.5, -0.666, -2.235]
    LARd = np.zeros(8)
    for i in range(8):
        LARd[i] = (LARc[i] - B[i]) / A[i]

    # Interpolation of the log-Area Rations (optional)

    # Transformation of the log-Area Rations into reflection coefficients
    rd = np.zeros(8)
    if np.abs(LARd[i]) < 0.675:
        rd[i] = LARd[i]
    elif np.abs(LARd[i] >= 0.675 and np.abs(LARd[i] < 1.225)):
        rd[i] = np.sign(LARd[i]) * (0.5 * np.abs(LARd[i] + 0.3375))
    elif np.abs(LARd[i] >= 1.225 and np.abs(LARd[i]) <= 1.625):
        rd[i] = np.sign(LARd[i]) * (0.125 * np.abs(LARd[i] + 0.796875))

    # use existing function from hw_utils to convert reflection coeffs to poly coeffs
    a, e_final = reflection_coeff_to_polynomial_coeff(rd)

    # get residual for current frame
    d = curr_frame_st_resd

    # FIR filter coefficients
    coeffs = [1, -a[0], -a[1], -a[2], -a[3], -a[4], -a[5], -a[6], -a[7]]

    # calculate decoded signal using FIR filter
    s_dec = lfilter(1, coeffs, d)

    # post-processing
    beta = 28180 * 2 ** (-15)
    s_ro = np.zeros(160)
    for i in range(1, 160):
        s_ro[i] = s_dec[i] + beta * s_ro[i - 1]

    return s_ro

    def RPE_frame_slt_decoder(LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd):
        d_synth = curr_frame_st_resd
        e = curr_frame_ex_full

        # decoding of Nc values
        N = Nc

        # decoding of bc values
        QLB = [0.1, 0.35, 0.65, 1]
        for j in range(4):
            b[j] = QLB[bc[j]]

        # Long term synthesis filtering
        for j in range(4):
            for k in range(40):
                d_synth[k0 + j * 40 + k] = (
                    e[k0 + j * 40 + k] + b[j] * d_est[k0 + j * 40 + k]
                )
                d_est[k] = d_synth[k - N[j]]

    return s0, curr_frame_st_resd
