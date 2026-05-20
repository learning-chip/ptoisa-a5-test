#!/usr/bin/env python3
"""Generate main.cpp for elementwise binary-op tests (tmax, tmul)."""

import sys
from pathlib import Path

TEMPLATE = r'''/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "test_runner.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace PtoTestCommon;
using namespace PtoTestRunner;

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols, bool sameTile>
void Launch{OP}(T *out, T *src0, T *src1, void *stream);

template <int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows, int vCols,
          bool sameTile>
void Launch{OP}Half(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols, bool isHalf = false,
          bool sameTile = (dstTileH == src0TileH && dstTileH == src1TileH && dstTileW == src0TileW &&
                           dstTileW == src1TileW)>
bool Run{OP}Case(const std::string &dataDir)
{{
    size_t fileSizeDst = dstTileH * dstTileW * sizeof(T);
    size_t fileSizeSrc0 = src0TileH * src0TileW * sizeof(T);
    size_t fileSizeSrc1 = src1TileH * src1TileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost = nullptr;
    T *src0Host = nullptr;
    T *src1Host = nullptr;
    T *dstDevice = nullptr;
    T *src0Device = nullptr;
    T *src1Device = nullptr;

    aclrtMallocHost((void **)(&dstHost), fileSizeDst);
    aclrtMallocHost((void **)(&src0Host), fileSizeSrc0);
    aclrtMallocHost((void **)(&src1Host), fileSizeSrc1);
    aclrtMalloc((void **)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSizeSrc1, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    if (!ReadFile(dataDir + "/input1.bin", readSize, src0Host, fileSizeSrc0)) {{
        return false;
    }}
    if (!ReadFile(dataDir + "/input2.bin", readSize, src1Host, fileSizeSrc1)) {{
        return false;
    }}

    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSizeSrc1, src1Host, fileSizeSrc1, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (isHalf) {{
        Launch{OP}Half<dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols, sameTile>(
            dstDevice, src0Device, src1Device, stream);
    }} else {{
        Launch{OP}<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols, sameTile>(
            dstDevice, src0Device, src1Device, stream);
    }}

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(dataDir + "/output.bin", dstHost, fileSizeDst);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSizeDst / sizeof(T));
    std::vector<T> devFinal(fileSizeDst / sizeof(T));
    readSize = 0;
    if (!ReadFile(dataDir + "/golden.bin", readSize, golden.data(), fileSizeDst)) {{
        return false;
    }}
    if (!ReadFile(dataDir + "/output.bin", readSize, devFinal.data(), fileSizeDst)) {{
        return false;
    }}
    return ResultCmp(golden, devFinal, 0.001f);
}}

#define {OP}_CASE(name, ...) {{name, Run{OP}Case<__VA_ARGS__>}}

static const std::vector<CaseSpec> kSmokeCases = {{
    {OP}_CASE("case_float_16x32_16x64_16x32_16x32", float, 16, 32, 16, 64, 16, 32, 16, 32),
    {OP}_CASE("case_int32_16x32_16x64_16x32_16x32", int32_t, 16, 32, 16, 64, 16, 32, 16, 32),
}};

#ifdef PTOISA_FULL_TEST
static const std::vector<CaseSpec> kFullCases = {{
    {OP}_CASE("case_float_64x64_64x64_64x64_64x64", float, 64, 64, 64, 64, 64, 64, 64, 64),
    {OP}_CASE("case_int32_64x64_64x64_64x64_64x64", int32_t, 64, 64, 64, 64, 64, 64, 64, 64),
    {OP}_CASE("case_int16_64x64_64x64_64x64_64x64", int16_t, 64, 64, 64, 64, 64, 64, 64, 64),
    {OP}_CASE("case_half_16x256_16x256_16x256_16x256", aclFloat16, 16, 256, 16, 256, 16, 256, 16, 256, true),
    {OP}_CASE("case_half_16x64_16x128_16x128_16x64", aclFloat16, 16, 64, 16, 128, 16, 128, 16, 64, true),
    {OP}_CASE("case_int16_32x128_32x128_32x256_32x128", int16_t, 32, 128, 32, 128, 32, 256, 32, 128),
    {OP}_CASE("case_half_16x64_16x128_16x128_16x63", aclFloat16, 16, 64, 16, 128, 16, 128, 16, 63, true),
    {OP}_CASE("case_float_16x32_16x64_16x32_16x31", float, 16, 32, 16, 64, 16, 32, 16, 31),
    {OP}_CASE("case_int16_32x128_32x128_32x256_32x127", int16_t, 32, 128, 32, 128, 32, 256, 32, 127),
    {OP}_CASE("case_int32_16x32_16x64_16x32_16x31", int32_t, 16, 32, 16, 64, 16, 32, 16, 31),
}};
#endif

int main(int argc, char **argv)
{{
#ifdef PTOISA_FULL_TEST
    return MainImpl(argc, argv, kSmokeCases, kFullCases);
#else
    return MainImpl(argc, argv, kSmokeCases);
#endif
}}
'''

def main():
    op = sys.argv[1]  # TMax or TMul
    out = Path(sys.argv[2])
    out.write_text(TEMPLATE.format(OP=op))

if __name__ == "__main__":
    main()
