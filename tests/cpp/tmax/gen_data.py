#!/usr/bin/env python3
# coding=utf-8

import argparse
import os

import numpy as np

np.random.seed(19)


class TMaxParams:
    def __init__(self, name, dtype, dstH, dstW, src0H, src0W, src1H, src1W, vRow, vCol):
        self.name = name
        self.dtype = dtype
        self.dst_tile_row = dstH
        self.dst_tile_col = dstW
        self.src0_tile_row = src0H
        self.src0_tile_col = src0W
        self.src1_tile_row = src1H
        self.src1_tile_col = src1W
        self.valid_row = vRow
        self.valid_col = vCol


def gen_golden_data(param):
    dtype = param.dtype
    input1 = np.random.randint(1, 10, size=[param.src0_tile_row, param.src0_tile_col]).astype(dtype)
    input2 = np.random.randint(1, 10, size=[param.src1_tile_row, param.src1_tile_col]).astype(dtype)
    golden = np.zeros([param.dst_tile_row, param.dst_tile_col]).astype(dtype)
    golden[0 : param.valid_row, 0 : param.valid_col] = np.maximum(
        input1[0 : param.valid_row, 0 : param.valid_col], input2[0 : param.valid_row, 0 : param.valid_col]
    )
    input1.tofile("input1.bin")
    input2.tofile("input2.bin")
    golden.tofile("golden.bin")


SMOKE_CASES = [
    TMaxParams("case_float_16x32_16x64_16x32_16x32", np.float32, 16, 32, 16, 64, 16, 32, 16, 32),
    TMaxParams("case_int32_16x32_16x64_16x32_16x32", np.int32, 16, 32, 16, 64, 16, 32, 16, 32),
]

FULL_CASES = [
    TMaxParams("case_float_64x64_64x64_64x64_64x64", np.float32, 64, 64, 64, 64, 64, 64, 64, 64),
    TMaxParams("case_int32_64x64_64x64_64x64_64x64", np.int32, 64, 64, 64, 64, 64, 64, 64, 64),
    TMaxParams("case_int16_64x64_64x64_64x64_64x64", np.int16, 64, 64, 64, 64, 64, 64, 64, 64),
    TMaxParams("case_half_16x256_16x256_16x256_16x256", np.float16, 16, 256, 16, 256, 16, 256, 16, 256),
    TMaxParams("case_half_16x64_16x128_16x128_16x64", np.float16, 16, 64, 16, 128, 16, 128, 16, 64),
    TMaxParams("case_int16_32x128_32x128_32x256_32x128", np.int16, 32, 128, 32, 128, 32, 256, 32, 128),
    TMaxParams("case_half_16x64_16x128_16x128_16x63", np.float16, 16, 64, 16, 128, 16, 128, 16, 63),
    TMaxParams("case_float_16x32_16x64_16x32_16x31", np.float32, 16, 32, 16, 64, 16, 32, 16, 31),
    TMaxParams("case_int16_32x128_32x128_32x256_32x127", np.int16, 32, 128, 32, 128, 32, 256, 32, 127),
    TMaxParams("case_int32_16x32_16x64_16x32_16x31", np.int32, 16, 32, 16, 64, 16, 32, 16, 31),
]


def main():
    parser = argparse.ArgumentParser(description="Generate tmax golden data")
    parser.add_argument("--output-dir", default="cases")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cases = list(SMOKE_CASES)
    if args.all:
        cases.extend(FULL_CASES)

    os.makedirs(args.output_dir, exist_ok=True)
    for param in cases:
        case_dir = os.path.join(args.output_dir, param.name)
        os.makedirs(case_dir, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(case_dir)
        gen_golden_data(param)
        os.chdir(cwd)
        print(f"Generated {case_dir}")


if __name__ == "__main__":
    main()
