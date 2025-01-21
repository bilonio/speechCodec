import numpy as np
import sys

sys.path.append("./material")
from hw_utils import reflection_coeff_to_polynomial_coeff
from scipy.signal import lfilter
from utils import decoding_coeff


def RPE_frame_st_decoder(LARc, curr_frame_st_resd):
    # Decoding of the quantized Log-Area ratios
    A = [20, 20, 20, 20, 13.637, 15.00, 8.334, 8.824]
    B = [0, 0, 4, -5, 0.184, -3.5, -0.666, -2.235]

    # get residual for current frame
    d = curr_frame_st_resd

    # decode LARc
    coeffs = decoding_coeff(LARc, A, B)

    # calculate decoded signal using FIR filter
    s_dec = lfilter(coeffs, 1, d)

    # post-processing
    beta = 28180 * (2 ** (-15))
    s_ro = np.zeros(160)
    s_ro = lfilter([1], [1, -beta], s_dec)

    return s_ro


def RPE_frame_slt_decoder(LARc, Nc, bc, curr_frame_ex_full, prev_frame_st_resd):

    e = curr_frame_ex_full
    # decoding of Nc values
    N = Nc

    # decoding of bc values
    b = np.zeros(4)
    QLB = [0.1, 0.35, 0.65, 1]
    for j in range(4):
        b[j] = QLB[bc[j]]

    d_synth = np.zeros(160)
    # Long term synthesis filtering
    for j in range(4):
        for i in range(40):
            d_synth[i + 40 * j] = (
                e[i + 40 * j] + b[j] * prev_frame_st_resd[i + 40 * j - N[j] + 159]
            )
            prev_frame_st_resd.append(d_synth[i + 40 * j])

    s_ro = RPE_frame_st_decoder(LARc, d_synth)

    return s_ro
