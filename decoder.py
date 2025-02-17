import numpy as np
import sys

sys.path.append("./material")
from hw_utils import reflection_coeff_to_polynomial_coeff
from scipy.signal import lfilter
from utils import decoding_coeff, LARc_from_bits, get_bits, read_bins


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
    prev_frame_st_resd = prev_frame_st_resd.tolist()

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


def RPE_frame_decoder(frame_bit_stream, prev_frame_st_resd):
    prev_frame_st_resd = prev_frame_st_resd.tolist()  # convert to list
    # decode the bitstream
    LARc = LARc_from_bits(frame_bit_stream)
    d_synth = np.zeros(160)
    decoded_x_mc_values = np.array(
        [-28672, -20480, -12288, -4096, 4096, 12288, 20480, 28672]
    ) * (2 ** (-15))
    decoded_x_mc = np.zeros(13)
    decoded_xm = np.zeros(13)
    e_deq = np.zeros(40)
    Nc = np.zeros(4)
    bc = np.zeros(4)
    Mc = np.zeros(4)
    x_mc = np.zeros(13)
    b = np.zeros(4)
    N = np.zeros(4)
    QLB = [0.1, 0.35, 0.65, 1]
    bins = read_bins()
    for j in range(4):
        Nc[j] = get_bits(frame_bit_stream, 36 + j * 56, 7).uint
        bc[j] = get_bits(frame_bit_stream, 43 + j * 56, 2).uint
        Mc[j] = get_bits(frame_bit_stream, 45 + 56 * j, 2).uint
        x_maxc = get_bits(frame_bit_stream, 47 + 56 * j, 6).uint
        x_max_deq = bins[x_maxc]
        b[j] = QLB[int(bc[j])]
        for i in range(13):
            x_mc[i] = get_bits(frame_bit_stream, 53 + 56 * j + 3 * i, 3).uint
            decoded_x_mc[i] = decoded_x_mc_values[int(x_mc[i])]
            decoded_xm[i] = decoded_x_mc[i] * x_max_deq
            e_deq[i * 3 + int(Mc[j])] = decoded_xm[i]

        N[j] = Nc[j]
        for i in range(40):
            d_synth[i + 40 * j] = (
                e_deq[i] + b[j] * prev_frame_st_resd[i + 40 * j - int(N[j]) + 159]
            )
            prev_frame_st_resd.append(d_synth[i + 40 * j])

    s_ro = RPE_frame_st_decoder(LARc, d_synth)

    return s_ro, d_synth
