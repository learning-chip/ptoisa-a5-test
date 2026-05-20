/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "test_runner.hpp"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace PtoTestCommon;
using namespace PtoTestRunner;

void LaunchHardSyncAll(int32_t *out, int32_t *flags, int32_t totalBlocks, void *stream);

bool RunHardAivCase(const std::string &)
{
    constexpr int32_t blockCount = 18;
    constexpr size_t int32PerCacheLine = 8;
    constexpr size_t elementCount = blockCount * int32PerCacheLine;
    constexpr size_t byteSize = elementCount * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *outHost = nullptr;
    int32_t *flagsHost = nullptr;
    int32_t *outDevice = nullptr;
    int32_t *flagsDevice = nullptr;

    aclrtMallocHost(reinterpret_cast<void **>(&outHost), byteSize);
    aclrtMallocHost(reinterpret_cast<void **>(&flagsHost), byteSize);
    aclrtMalloc(reinterpret_cast<void **>(&outDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&flagsDevice), byteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    std::fill_n(outHost, elementCount, 0);
    std::fill_n(flagsHost, elementCount, 0);
    aclrtMemcpy(outDevice, byteSize, outHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(flagsDevice, byteSize, flagsHost, byteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchHardSyncAll(outDevice, flagsDevice, blockCount, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, byteSize, outDevice, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<int32_t> golden(blockCount);
    std::vector<int32_t> devFinal(blockCount);
    for (size_t i = 0; i < blockCount; ++i) {
        golden[i] = 1;
        devFinal[i] = outHost[i * int32PerCacheLine];
    }

    const bool ok = ResultCmp(golden, devFinal, 0.0f);

    aclrtFree(outDevice);
    aclrtFree(flagsDevice);
    aclrtFreeHost(outHost);
    aclrtFreeHost(flagsHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return ok;
}

static const std::vector<CaseSpec> kSmokeCases = {
    {"case_hard_aiv_only_all_blocks", RunHardAivCase},
};

int main(int argc, char **argv)
{
    return MainImpl(argc, argv, kSmokeCases);
}
