/**
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
#include <fstream>
#include <string>
#include <vector>

using namespace PtoTestCommon;
using namespace PtoTestRunner;

template <int32_t testKey>
void launchTLOAD(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);

template <int32_t testKey>
int get_input_golden(uint8_t *input, uint8_t *golden);

constexpr int kLogSize = 128;
constexpr int kMaxBlock = 64;

template <int32_t testKey, typename T, int32_t kBlock>
bool RunTLoadCase(const std::string &dataDir)
{
    (void)dataDir;
    constexpr uint32_t M = 1024;
    constexpr uint32_t N = 1024;
    constexpr int inByteSize = M * N * static_cast<int>(sizeof(float));
    constexpr int outByteSize = M * N * static_cast<int>(sizeof(float));

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    void *dstHost = nullptr;
    void *srcHost = nullptr;
    void *goldHost = nullptr;
    void *dstDevice = nullptr;
    void *srcDevice = nullptr;
    void *logDevice = nullptr;

    aclrtMallocHost(&dstHost, outByteSize);
    aclrtMallocHost(&srcHost, inByteSize);
    aclrtMallocHost(&goldHost, outByteSize);
    aclrtMalloc(&dstDevice, inByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&srcDevice, outByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    const int actualOutByteSize =
        get_input_golden<testKey>(static_cast<uint8_t *>(srcHost), static_cast<uint8_t *>(goldHost));
    std::fill(static_cast<uint8_t *>(dstHost), static_cast<uint8_t *>(dstHost) + outByteSize, 0);

    aclrtMemcpy(srcDevice, inByteSize, srcHost, inByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, outByteSize, dstHost, outByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t logHost[kMaxBlock][kLogSize];
    std::fill(reinterpret_cast<uint8_t *>(logHost), reinterpret_cast<uint8_t *>(logHost) + sizeof(logHost), 0);
    aclrtMalloc(&logDevice, sizeof(logHost), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(logDevice, sizeof(logHost), logHost, sizeof(logHost), ACL_MEMCPY_HOST_TO_DEVICE);

    launchTLOAD<testKey>(static_cast<uint8_t *>(dstDevice), static_cast<uint8_t *>(srcDevice),
                         static_cast<uint64_t *>(logDevice), stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outByteSize, dstDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    const int elements = actualOutByteSize / static_cast<int>(sizeof(T));
    std::vector<T> golden(elements);
    std::vector<T> devFinal(elements);
    std::memcpy(golden.data(), goldHost, actualOutByteSize);
    std::memcpy(devFinal.data(), dstHost, actualOutByteSize);
    const bool ok = ResultCmp(golden, devFinal, 0.0f);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(logDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(goldHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return ok;
}

#define TLOAD_CASE(name, key, type, block) \
    { name, RunTLoadCase<key, type, block> }

static const std::vector<CaseSpec> kSmokeCases = {
    TLOAD_CASE("case_float_GT_128_128_VT_128_128_BLK1", 1, float, 1),
};

#ifdef PTOISA_FULL_TEST
static const std::vector<CaseSpec> kFullCases = {
    TLOAD_CASE("case_float_GT_2_2_2_256_64_VT_256_64_BLK8", 2, float, 8),
    TLOAD_CASE("case_float_GT_128_127_VT_128_128_BLK1_PADMAX", 3, float, 1),
    TLOAD_CASE("case_s16_GT_128_127_VT_128_128_BLK1_PADMAX", 4, int16_t, 1),
    TLOAD_CASE("case_u8_GT_128_127_VT_128_128_BLK1_PADMIN", 5, uint8_t, 1),
    TLOAD_CASE("case_float_GT_32_64_128_VT_64_128_BLK32_DYN", 6, int16_t, 32),
    TLOAD_CASE("case_float_GT_32_64_128_VT_64_128_BLK32_STC", 7, int16_t, 32),
    TLOAD_CASE("case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX", 8, float, 8),
    TLOAD_CASE("case_int64_GT_128_128_VT_128_128_BLK1", 9, int64_t, 1),
    TLOAD_CASE("case_uint64_GT_128_125_VT_128_128_BLK1_PADZERO", 10, uint64_t, 1),
    TLOAD_CASE("case_int64_GT_2_2_2_256_62_VT_256_64_BLK8_PADZERO", 11, int64_t, 8),
    TLOAD_CASE("case_uint64_GT_2_2_2_256_64_VT_256_64_BLK8", 12, uint64_t, 8),
};
#endif

int main(int argc, char **argv)
{
#ifdef PTOISA_FULL_TEST
    return MainImpl(argc, argv, kSmokeCases, kFullCases);
#else
    return MainImpl(argc, argv, kSmokeCases);
#endif
}
