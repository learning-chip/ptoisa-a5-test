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

#include <cstring>
#include <string>
#include <vector>

using namespace PtoTestCommon;
using namespace PtoTestRunner;

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t funcKey, typename CType, typename AType, typename BType, int32_t key, uint32_t IdxRow,
          uint32_t IdxCol, bool isInsert, uint32_t DstRow, uint32_t DstCol, uint32_t M, uint32_t K, uint32_t N>
bool RunExtractCase(const std::string &dataDir)
{
    size_t aFileSize = M * K * sizeof(AType);
    size_t bFileSize = K * N * sizeof(BType);
    size_t cFileSize = DstRow * DstCol * sizeof(CType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost = nullptr;
    uint8_t *src0Host = nullptr;
    uint8_t *src1Host = nullptr;
    uint8_t *src2Host = nullptr;
    uint8_t *dstDevice = nullptr;
    uint8_t *src0Device = nullptr;
    uint8_t *src1Device = nullptr;
    uint8_t *src2Device = nullptr;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), cFileSize);
    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    if (!ReadFile(dataDir + "/x1_gm.bin", readSize, src0Host, aFileSize)) {
        return false;
    }
    if (!ReadFile(dataDir + "/x2_gm.bin", readSize, src1Host, bFileSize)) {
        return false;
    }
    if (!ReadFile(dataDir + "/dst.bin", readSize, src2Host, cFileSize)) {
        return false;
    }

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, cFileSize, src2Host, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (funcKey == 1) {
        LaunchTMOVAcc2VecNZ2ND<key>(dstDevice, src0Device, src1Device, src2Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(dataDir + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<CType> golden(cFileSize / sizeof(CType));
    std::vector<CType> devFinal(cFileSize / sizeof(CType));
    readSize = 0;
    if (!ReadFile(dataDir + "/golden.bin", readSize, golden.data(), cFileSize)) {
        return false;
    }
    if (!ReadFile(dataDir + "/output_z.bin", readSize, devFinal.data(), cFileSize)) {
        return false;
    }
    return ResultCmp(golden, devFinal, 0.001f);
}

static const std::vector<CaseSpec> kSmokeCases = {
    {"case_nz2nd_3",
     RunExtractCase<1, uint32_t, uint16_t, uint16_t, 3, 2, 0, false, 10, 16, 6, 7, 8>},
};

int main(int argc, char **argv)
{
    return MainImpl(argc, argv, kSmokeCases);
}
