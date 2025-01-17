import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import lfilter
import sys

sys.path.append("./material")
from hw_utils import (
    polynomial_coeff_to_reflection_coeff,
    reflection_coeff_to_polynomial_coeff,
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
    alpha = 32735 * 2 ** (-15)
    for i in range(1, 160):
        s_of[i] = s0[i] - s0[i - 1] + alpha * s_of[i - 1]

    # Preemphasis
    s = np.zeros(160)  # preemphasized signal
    beta = 28180 * 2 ** (-15)
    for i in range(1, 160):
        s[i] = s_of[i] - beta * s_of[i - 1]
    # Short term analysis
    # LPC analysis section

    # Autocorrelation
    ACF = np.zeros(9)  # autocorrelation coefficients
    for k in range(9):
        ACF[k] = sum([s[j] * s[j - k] for j in range(k, 160)])

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
    r = polynomial_coeff_to_reflection_coeff(w)

    # Transformation of reflection coefficients to LAR (Log Area Ratios)
    for i in range(0, len(r)):
        if np.abs(r[i]) < 0.675:
            LAR[i] = r[i]
        elif np.abs(r[i] >= 0.675 and np.abs(r[i]) < 0.95):
            LAR[i] = np.sign(r[i]) * (2 * np.abs(r[i]) - 0.675)
        elif np.abs(r[i] >= 0.95 and np.abs(r[i]) <= 1.0):
            LAR[i] = np.sign(r[i]) * (8 * np.abs(r[i]) - 0.6375)

    # Quantization of LAR
    LARc = np.zeros(8)  # quantized LAR
    A = [20, 20, 20, 20, 13.637, 15.00, 8.334, 8.824]
    B = [0, 0, 4, -5, 0.184, -3.5, -0.666, -2.235]
    for i in range(8):
        z = A[i] * LAR[i] + B[i]
        LARc[i] = int(z + np.sign(z) * 0.5)

    # short-term analysis filtering section

    # Decoding of the quantized Log-Area Rations
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

    a, e_final = reflection_coeff_to_polynomial_coeff(rd)
    # Short term analysis filtering
    coeff = [1, -a[0], -a[1], -a[2], -a[3], -a[4], -a[5], -a[6], -a[7]]
    d = lfilter(coeff, 1, s)  # apply FIR filter
    return LARc, d


def RPE_frame_slt_coder(s0, prev_frame_st_resd):
    # Preprocessing

    # Offset compensation
    s_of = np.zeros(160)  # offset compensated signal
    alpha = 32735 * 2 ** (-15)
    for i in range(1, 160):
        s_of[i] = s0[i] - s0[i - 1] + alpha * s_of[i - 1]

    # Preemphasis
    s = np.zeros(160)  # preemphasized signal
    beta = 28180 * 2 ** (-15)
    for i in range(1, 160):
        s[i] = s_of[i] - beta * s_of[i - 1]
    # Short term analysis
    # LPC analysis section

    # Autocorrelation
    ACF = np.zeros(9)  # autocorrelation coefficients
    for k in range(9):
        ACF[k] = sum([s[j] * s[j - k] for j in range(k, 160)])

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
    r = polynomial_coeff_to_reflection_coeff(w)

    # Transformation of reflection coefficients to LAR (Log Area Ratios)
    for i in range(0, len(r)):
        if np.abs(r[i]) < 0.675:
            LAR[i] = r[i]
        elif np.abs(r[i] >= 0.675 and np.abs(r[i]) < 0.95):
            LAR[i] = np.sign(r[i]) * (2 * np.abs(r[i]) - 0.675)
        elif np.abs(r[i] >= 0.95 and np.abs(r[i]) <= 1.0):
            LAR[i] = np.sign(r[i]) * (8 * np.abs(r[i]) - 0.6375)

    # Quantization of LAR
    LARc = np.zeros(8)  # quantized LAR
    A = [20, 20, 20, 20, 13.637, 15.00, 8.334, 8.824]
    B = [0, 0, 4, -5, 0.184, -3.5, -0.666, -2.235]
    for i in range(8):
        z = A[i] * LAR[i] + B[i]
        LARc[i] = int(z + np.sign(z) * 0.5)

    # short-term analysis filtering section

    # Decoding of the quantized Log-Area Rations
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

    a, e_final = reflection_coeff_to_polynomial_coeff(rd)
    # Short term analysis filtering
    coeff = [1, -a[0], -a[1], -a[2], -a[3], -a[4], -a[5], -a[6], -a[7]]
    d = lfilter(coeff, 1, s)  # apply FIR filter

    # Long term analysis

    # Calculation of the LTP parameters
   # prev_d = prev_frame_st_resd[40:160]  # extract only the last 120 samples
    prev_d = prev_frame_st_resd
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

        # Long term analysis filtering
        for k in range(1, 40):
            # print(prev_d[kj - Nc[j]])
            e[kj + k] = d[kj + k] - bc[j] * prev_d[119 + k - Nc[j]]
            prev_d.append(e[kj + k] + b[j] * prev_d[119 + k - N[j]])
            prev_d.pop(0)

    return LARc, Nc, bc, e, prev_d


def RPE_subframe_slt_lte(d, prev_d):
    # prev_d = np.concatenate(prev_d)
    R = np.zeros(80)
    S = np.zeros(80)
    for i in range(40):
        for lamda in range(40, 120):
            # print("dokimh", i, lamda, prev_d[i+119 - lamda],d[i])
            R[119 - lamda] += d[i] * prev_d[i + 119 - lamda]
            S[119 - lamda] += (prev_d[i + 119 - lamda]) ** 2
    N = np.argmax(R)
    b = R[N] / S[N]
    #print("N =", 120 - N)
    return 120 - N, b
