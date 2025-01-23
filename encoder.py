import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter
from bitstring import BitArray
import sys

sys.path.append("./material")
from hw_utils import (
    polynomial_coeff_to_reflection_coeff,
    reflection_coeff_to_polynomial_coeff,
)
from utils import (
    decoding_coeff,
    quantize_LAR,
    set_bits,
    LARc_bits,
    write_bins_to_csv,
    read_bins,
    quantize_values,
)


def RPE_frame_st_coder(s0, prev_frame_st_resd):
    # s0: current frame
    # prev_frame_st_resd: residual of previous frame
    # LARc: Linear Prediction Coefficients
    # curr_frame_st_resd: residual of current frame
    # return LARc, curr_frame_st_resd

    # Preprocessing

    # Offset compensation
    s_of = np.zeros(160)  # offset compensated signal
    s_of[0] = s0[0] - prev_frame_st_resd[-1] if len(prev_frame_st_resd) > 0 else s0[0]
    alpha = 32735 * (2 ** (-15))
    for i in range(1, 160):
        s_of[i] = s0[i] - s0[i - 1] + alpha * s_of[i - 1]

    # Preemphasis
    s = np.zeros(160)  # preemphasized signal
    s[0] = s_of[0]
    beta = 28180 * (2 ** (-15))
    s = lfilter([1, -beta], [1], s_of)

    # Short term analysis
    # LPC analysis section

    # Autocorrelation
    ACF = np.zeros(9)  # autocorrelation coefficients
    for k in range(9):
        ACF[k] = sum([s[i] * s[i - k] for i in range(k, 160)])

    # Calculate reflection coefficients
    w = np.zeros(8)  # reflection coefficients
    LAR = np.zeros(8)  # Linear Prediction Coefficients
    R = np.zeros((8, 8))  # Prediction Coefficients

    # Construct the Toeplitz matrix R using the ACF values
    R = toeplitz(ACF[:8])  # Create an 8x8 Toeplitz matrix from ACF[0] to ACF[7]

    # Create the vector r using ACF values
    r = np.array(ACF[1:9])  # Use ACF[1] to ACF[8] for the r vector

    w = np.linalg.solve(R, r)

    # use existing function from hw_utils to convert poly coeff to reflection coefficients
    r = np.zeros(8)
    r = polynomial_coeff_to_reflection_coeff(w)

    # Transformation of reflection coefficients to LAR (Log Area Ratios)
    for i in range(0, len(r)):
        if np.abs(r[i]) < 0.675:
            LAR[i] = r[i]
        elif 0.675 <= np.abs(r[i]) < 0.95:
            LAR[i] = np.sign(r[i]) * (2 * np.abs(r[i]) - 0.675)
        elif 0.95 <= np.abs(r[i]) <= 1.0:
            LAR[i] = np.sign(r[i]) * (8 * np.abs(r[i]) - 6.375)

    LARc = np.zeros(8)  # quantized LAR
    LARd = np.zeros(8)  # decoded LAR
    rd = np.zeros(8)  # decoded reflection coefficients
    A = [20, 20, 20, 20, 13.637, 15.00, 8.334, 8.824]
    B = [0, 0, 4, -5, 0.184, -3.5, -0.666, -2.235]
    LARc_min = np.array([-32, -32, -16, -16, -8, -8, -4, -4])
    LARc_max = np.array([31, 31, 15, 15, 7, 7, 3, 3])

    # Quantization of LAR
    z = A * LAR + B
    LARc = [quantize_LAR(z[i], i) for i in range(8)]

    # Decoding of the quantized Log-Area Rations
    coeffs = decoding_coeff(LARc, A, B)

    d = lfilter(coeffs, 1, s)  # apply FIR filter
    return LARc, d


def RPE_frame_slt_coder(s0, prev_frame_st_resd):
    # short term analysis
    LARc, d = RPE_frame_st_coder(s0, prev_frame_st_resd)

    # Long term analysis

    # Calculation of the LTP parameters
    # prev_d = prev_frame_st_resd[40:160]  # extract only the last 120 samples
    prev_d = prev_frame_st_resd[40:160]  # extract only the last 120 samples
    # print(prev_d)
    k0 = 0
    DLB = [0.2, 0.5, 0.8]
    QLB = [0.1, 0.35, 0.65, 1]
    N = np.zeros(4, dtype=int)
    b = np.zeros(4)
    Nc = np.zeros(4, dtype=int)
    bc = np.zeros(4, dtype=int)
    e = np.zeros(160)
    for j in range(4):
        kj = k0 + 40 * j  # start of subframe
        N[j], b[j] = RPE_subframe_slt_lte(d[kj : kj + 40], prev_d)  # get N and b values
        Nc[j] = N[j]  # quantization of LTP delays

        # quantization of LTP gains
        if b[j] <= DLB[0]:
            bc[j] = 0
        elif b[j] > DLB[0] and b[j] <= DLB[1]:
            bc[j] = 1
        elif b[j] > DLB[1] and b[j] <= DLB[2]:
            bc[j] = 2
        elif b[j] > DLB[2]:
            bc[j] = 3

        # decoding of bc values
        b[j] = QLB[bc[j]]

        # Long term analysis filtering and synthesis filtering
        H = np.array(
            [-134, -374, 0, 2054, 5741, 8192, 5741, 2054, 0, -374, -134]
        )  # weighting filter
        H = H * (2 ** (-13))  # scale the filter
        x = np.zeros(40)  # output of the weighting filter

        for k in range(40):
            # print(prev_d[kj - Nc[j]])
            e[kj + k] = d[kj + k] - bc[j] * prev_d[119 + k - Nc[j]]
            prev_d.append(e[kj + k] + b[j] * prev_d[119 + k - N[j]])
            if j != 3:
                prev_d.pop(0)  # remove the first element

    return LARc, Nc, bc, e, prev_d


def RPE_frame_coder(s0, prev_frame_st_resd):
    bits = BitArray(length=260)
    # short term analysis
    LARc, d = RPE_frame_st_coder(s0, prev_frame_st_resd)

    LARc_bits(bits, LARc)

    # Long term analysis

    # Calculation of the LTP parameters
    # prev_d = prev_frame_st_resd[40:160]  # extract only the last 120 samples
    prev_d = prev_frame_st_resd[40:160]  # extract only the last 120 samples
    # print(prev_d)
    k0 = 0
    DLB = [0.2, 0.5, 0.8]
    QLB = [0.1, 0.35, 0.65, 1]
    N = np.zeros(4, dtype=int)
    b = np.zeros(4)
    Nc = np.zeros(4, dtype=int)
    bc = np.zeros(4, dtype=int)
    e = np.zeros(160)
    for j in range(4):
        kj = k0 + 40 * j  # start of subframe
        N[j], b[j] = RPE_subframe_slt_lte(d[kj : kj + 40], prev_d)  # get N and b values
        Nc[j] = N[j]  # quantization of LTP delays

        # quantization of LTP gains
        if b[j] <= DLB[0]:
            bc[j] = 0
        elif b[j] > DLB[0] and b[j] <= DLB[1]:
            bc[j] = 1
        elif b[j] > DLB[1] and b[j] <= DLB[2]:
            bc[j] = 2
        elif b[j] > DLB[2]:
            bc[j] = 3

        set_bits(bits, 36 + j * 56, 7, Nc[j])
        # print(bits[36 + j*56: 36 + j*56 + 7])
        set_bits(bits, 43 + j * 56, 2, bc[j])

        # decoding of bc values
        b[j] = QLB[bc[j]]

        # Long term analysis filtering and synthesis filtering
        H = np.array(
            [-134, -374, 0, 2054, 5741, 8192, 5741, 2054, 0, -374, -134]
        )  # weighting filter
        H = H * (2 ** (-13))  # scale the filter
        x = np.zeros(40)  # output of the weighting filter

        for k in range(40):
            e[kj + k] = d[kj + k] - bc[j] * prev_d[119 + k - Nc[j]]
            x[k] = sum(H[i] * e[k + 5 - i] for i in range(11))

        xm = np.zeros((4, 13))
        E = np.zeros(4)
        for m in range(4):
            for i in range(13):
                xm[m][i] = x[m + 3 * i]

            E[m] = sum(xm[m][i] ** 2 for i in range(13))
        # print(E)
        M = np.argmax(E)  # find the grid position with the maximum energy
        Mc = M
        set_bits(bits, 45 + 56 * j, 2, Mc)
        bins = read_bins()  # read the bins for quantization
        x_rpe = xm[M]  # RPE sequence is the one with the maximum energy
        x_max = max(np.abs(x_rpe))  # find the maximum value of the RPE sequence
        x_maxc, x_max_deq = quantize_values(x_max, bins)  # quantize the maximum value
        set_bits(bits, 47 + 56 * j, 6, x_maxc)
        x_deq = x_rpe / x_max_deq  # normalize the RPE sequence
        normalized_rpe_bins = np.array(
            [-24577, -16385, -8193, -1, 8191, 16383, 24575, 32767]
        )
        normalized_rpe_bins = normalized_rpe_bins * (2 ** (-15))
        x_mc = np.zeros(13)
        for i in range(13):
            x_mc[i], _ = quantize_values(
                x_deq[i], normalized_rpe_bins
            )  # quantize the normalized RPE sequence
            set_bits(bits, 53 + 56 * j + 3 * i, 3, x_mc[i])

        decoded_x_mc_values = np.array(
            [-28672, -20480, -12288, -4096, 4096, 12288, 20480, 28672]
        ) * (2 ** (-15))
        decoded_x_mc = np.zeros(13)
        decoded_xm = np.zeros(13)
        for i in range(13):
            decoded_x_mc[i] = decoded_x_mc_values[int(x_mc[i])]
            decoded_xm[i] = decoded_x_mc[i] * x_max_deq
        e_deq = np.zeros(40)

        for i in range(13):
            e_deq[i * 3 + Mc] = decoded_xm[i]

        for k in range(40):
            prev_d.append(e_deq[k] + b[j] * prev_d[119 + k - N[j]])
            if j != 3:
                prev_d.pop(0)  # remove the first element

    return bits, prev_d


def RPE_subframe_slt_lte(d, prev_d):
    R = np.zeros(80)
    S = np.zeros(80)
    for i in range(40):
        for lamda in range(40, 120):
            # print("dokimh", i, lamda, prev_d[i+119 - lamda],d[i])
            R[119 - lamda] += d[i] * prev_d[i + 119 - lamda]
            S[119 - lamda] += (prev_d[i + 119 - lamda]) ** 2
    N = np.argmax(R)

    # Calculate b
    if S[N] != 0:
        b = R[N] / S[N]
    else:
        b = 0  # if S[N] is zero for the first subframes
    return 120 - N, b
