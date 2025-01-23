import numpy as np
import csv
import sys
from bitstring import BitArray

sys.path.append("./material")
from hw_utils import (
    polynomial_coeff_to_reflection_coeff,
    reflection_coeff_to_polynomial_coeff,
)


def decoding_coeff(LARc, A, B):
    LARd = np.zeros(8)  # decoded LARc values
    rd = np.zeros(8)  # decoded reflection coefficients
    for i in range(8):
        LARd[i] = (LARc[i] - B[i]) / A[i]  # decoding of LARc values

        # Interpolation of the log-Area Rations (optional)

        # Transformation of the log-Area Rations into reflection coefficients
        if np.abs(LARd[i]) < 0.675:
            rd[i] = LARd[i]
        elif np.abs(LARd[i] >= 0.675 and np.abs(LARd[i] < 1.225)):
            rd[i] = np.sign(LARd[i]) * (0.5 * np.abs(LARd[i] + 0.3375))
        elif np.abs(LARd[i] >= 1.225 and np.abs(LARd[i]) <= 1.625):
            rd[i] = np.sign(LARd[i]) * (0.125 * np.abs(LARd[i] + 0.796875))

    a, e_final = reflection_coeff_to_polynomial_coeff(
        rd
    )  # convert reflection coeffs to poly coeffs

    coeffs = [
        a[0],
        -a[1],
        -a[2],
        -a[3],
        -a[4],
        -a[5],
        -a[6],
        -a[7],
        -a[8],
    ]  # FIR filter coefficients

    return coeffs


def quantize_LAR(z,index):
    LARc_min = np.array([-32, -32, -16, -16, -8, -8, -4, -4])
    LARc_max = np.array([31, 31, 15, 15, 7, 7, 3, 3])
    quantized_LAR = int(z + np.sign(z) * 0.5)  # quantization of z
    return max(LARc_min[index], min(LARc_max[index], quantized_LAR))

def set_bits(bits, start, length, value):
    # Calculate the signed and unsigned range for the given bit length
    signed_min = -(2 ** (length - 1))  # Minimum signed value (e.g., -16 for 5 bits)
    signed_max = (2 ** (length - 1)) - 1  # Maximum signed value (e.g., 15 for 5 bits)
    unsigned_max = (2 ** length) - 1  # Maximum unsigned value (e.g., 31 for 5 bits)

    if signed_min <= value <= signed_max:
        # Value fits in signed range
        bits[start:start + length] = BitArray(int=value, length=length)
    elif 0 <= value <= unsigned_max:
        # Value fits in unsigned range
        bits[start:start + length] = BitArray(uint=value, length=length)
    else:
        raise ValueError(f"Value {value} cannot fit in {length} bits.")
    
def get_bits(bits, start, length):
    return (bits[start:start + length])
        

def LARc_bits(bits, LARc):
    set_bits(bits, 0, 6, LARc[0])   # LAR 1 (bits 0-5)
    set_bits(bits, 6, 6, LARc[1])   # LAR 2 (bits 6-11)
    set_bits(bits, 12, 5, LARc[2])  # LAR 3 (bits 12-16)
    set_bits(bits, 17, 5, LARc[3])  # LAR 4 (bits 17-21)
    set_bits(bits, 22, 4, LARc[4])   # LAR 5 (bits 22-25)
    set_bits(bits, 26, 4, LARc[5])  # LAR 6 (bits 26-29)
    set_bits(bits, 30, 3, LARc[6])   # LAR 7 (bits 30-32)
    set_bits(bits, 33, 3, LARc[7])   # LAR 8 (bits 33-35)
    return bits

def LARc_from_bits(bits):
    LARc = np.zeros(8)
    LARc[0] = get_bits(bits, 0, 6).int  # LAR 1 (bits 0-5)
    LARc[1] = get_bits(bits, 6, 6).int  # LAR 2 (bits 6-11)
    LARc[2] = get_bits(bits, 12, 5).int  # LAR 3 (bits 12-16)
    LARc[3] = get_bits(bits, 17, 5).int  # LAR 4 (bits 17-21)
    LARc[4] = get_bits(bits, 22, 4).int  # LAR 5 (bits 22-25)
    LARc[5] = get_bits(bits, 26, 4).int  # LAR 6 (bits 26-29)
    LARc[6] = get_bits(bits, 30, 3).int  # LAR 7 (bits 30-32)
    LARc[7] = get_bits(bits, 33, 3).int  # LAR 8 (bits 33-35)
    return LARc

def write_bins_to_csv():
    bins = [
    31, 63, 95, 127, 159, 191, 223, 255, 287, 319, 351, 383, 415, 447, 479, 511,
    575, 639, 703, 767, 831, 895, 959, 1023, 1151, 1279, 1407, 1535, 1663, 1791,
    1919, 2047, 2303, 2559, 2815, 3071, 3327, 3583, 3839, 4095, 4607, 5119, 5631,
    6143, 6655, 7167, 7679, 8191, 9215, 10239, 11263, 12287, 13311, 14335, 15359,
    16383, 18431, 20479, 22527, 24575, 26623, 28671, 30719, 32767
    ]

    # Save to CSV file
    with open('bins.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(bins)
    

write_bins_to_csv()

def read_bins():
    # Read bins from CSV file
    with open('bins.csv', mode='r') as file:
        reader = csv.reader(file)
        bins = list(map(int, next(reader)))  # Convert strings to integers
    return bins

def quantize_values(x, bins):
    # Define the bins and their corresponding quantized values
    bin_edges = bins
    quantized_values = np.digitize(x, bins=bin_edges, right=True)
    x_edges = bin_edges[quantized_values]
    return quantized_values, x_edges

