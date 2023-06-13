import argparse
import ctypes
import numpy as np
import torch


def decompose(x: torch.Tensor, normalized: bool):
    """decomposes a torch float tensor into sign, exponent, and mantissa"""
    # Get the memory layout of x
    if x.dtype == torch.float32:
        m_bits = 23
        e_bits = 8
        e_bias = 127
    elif x.dtype == torch.bfloat16:
        m_bits = 7
        e_bits = 8
        e_bias = 127
    elif x.dtype == torch.float16:
        m_bits = 10
        e_bits = 5
        e_bias = 15
    else:
        raise ValueError("Unsupported arguments")
    data_type = ctypes.c_uint16 if x.element_size() == 2 else ctypes.c_uint32
    # Get the memory address of x
    data_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(data_type))
    # Load raw bytes at this address as a integer of the correct type
    int_array = np.ctypeslib.as_array(data_ptr, (x.numel(),))

    float_array = x.flatten()
    for i in range(int_array.size):
        # Get current item
        float_data = float_array[i]
        int_data = int_array[i]
        # Drop exponent and mantissa to get the sign
        sign = int_data >> (e_bits + m_bits)
        # Mask sign and drop mantissa to get the exponent, then remove its bias
        exponent = ((int_data & (2**(e_bits + m_bits) - 1)) >> m_bits) - e_bias
        # Mask sign and exponent to get the mantissa fraction, adding the implicit leading 1
        mantissa = (int_data & (2**m_bits - 1)) + 2 ** m_bits
        if normalized:
            # Divide by the precision to express mantissa as a normalized decimal number
            mantissa = mantissa.astype(np.float32) / 2. ** m_bits
        else:
            # Adjust exponent to take also into account the precision
            exponent -= m_bits
        # Dump representation
        bin = np.binary_repr(int_data, width=8 * x.element_size())
        print(f"Decomposing {float_data} represented as {bin} ({int_data})")
        print(sign, exponent, mantissa)
        print((1 - 2 * sign) * mantissa * 2. ** exponent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="float32", help="[float32|float16|bfloat16]")
    parser.add_argument("--normalized", action="store_true", help="Provide the mantissa as a normalized decimal number")
    parser.add_argument("number", type=float)
    args = parser.parse_args()

    dtype = getattr(torch, args.type)
    x = torch.tensor(args.number, dtype=dtype)
    decompose(x, args.normalized)