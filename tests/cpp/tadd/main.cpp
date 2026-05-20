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

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace PtoTestCommon;

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols>
void LaunchTAdd(T *out, T *src0, T *src1, void *stream);

template <int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows, int vCols>
void LaunchTAddHalf(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);

struct TAddCase {
    const char *name;
    bool (*run)(const std::string &dataDir);
};

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols, bool isHalf = false>
bool RunTAddCase(const std::string &dataDir)
{
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

    memset(dstHost, 0, fileSizeDst);
    size_t readSize = 0;
    if (!ReadFile(dataDir + "/input1.bin", readSize, src0Host, fileSizeSrc0)) {
        ERROR_LOG("Failed to read input1.bin for %s", dataDir.c_str());
        return false;
    }
    if (!ReadFile(dataDir + "/input2.bin", readSize, src1Host, fileSizeSrc1)) {
        ERROR_LOG("Failed to read input2.bin for %s", dataDir.c_str());
        return false;
    }

    aclrtMemcpy(dstDevice, fileSizeDst, dstHost, fileSizeDst, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSizeSrc1, src1Host, fileSizeSrc1, ACL_MEMCPY_HOST_TO_DEVICE);

    if constexpr (isHalf) {
        LaunchTAddHalf<dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(
            dstDevice, src0Device, src1Device, stream);
    } else {
        LaunchTAdd<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(
            dstDevice, src0Device, src1Device, stream);
    }

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
    if (!ReadFile(dataDir + "/golden.bin", readSize, golden.data(), fileSizeDst)) {
        ERROR_LOG("Failed to read golden.bin for %s", dataDir.c_str());
        return false;
    }
    if (!ReadFile(dataDir + "/output.bin", readSize, devFinal.data(), fileSizeDst)) {
        ERROR_LOG("Failed to read output.bin for %s", dataDir.c_str());
        return false;
    }

    return ResultCmp(golden, devFinal, 0.001f);
}

#define TADD_CASE(name, ...)                                                                                           \
    { name, RunTAddCase<__VA_ARGS__> }

static const TAddCase kSmokeCases[] = {
    TADD_CASE("case_float_64x64_64x64_64x64_64x64", float, 64, 64, 64, 64, 64, 64, 64, 64),
    TADD_CASE("case_int32_64x64_64x64_64x64_64x64", int32_t, 64, 64, 64, 64, 64, 64, 64, 64),
};

#ifdef PTOISA_FULL_TEST
static const TAddCase kFullCases[] = {
    TADD_CASE("case_float_64x128_64x128_64x128_64x128", float, 64, 128, 64, 128, 64, 128, 64, 128),
    TADD_CASE("case_int16_64x64_64x64_64x64_64x64", int16_t, 64, 64, 64, 64, 64, 64, 64, 64),
    TADD_CASE("case_half_16x256_16x256_16x256_16x256", aclFloat16, 16, 256, 16, 256, 16, 256, 16, 256, true),
    TADD_CASE("case_half_16x64_16x128_16x128_16x64", aclFloat16, 16, 64, 16, 128, 16, 128, 16, 64, true),
    TADD_CASE("case_float_16x32_16x64_16x32_16x32", float, 16, 32, 16, 64, 16, 32, 16, 32),
    TADD_CASE("case_int16_32x128_32x128_32x256_32x128", int16_t, 32, 128, 32, 128, 32, 256, 32, 128),
    TADD_CASE("case_int32_16x32_16x64_16x32_16x32", int32_t, 16, 32, 16, 64, 16, 32, 16, 32),
    TADD_CASE("case_half_16x64_16x128_16x128_16x63", aclFloat16, 16, 64, 16, 128, 16, 128, 16, 63, true),
    TADD_CASE("case_float_16x32_16x64_16x32_16x31", float, 16, 32, 16, 64, 16, 32, 16, 31),
    TADD_CASE("case_int16_32x128_32x128_32x256_32x127", int16_t, 32, 128, 32, 128, 32, 256, 32, 127),
    TADD_CASE("case_int32_16x32_16x64_16x32_16x31", int32_t, 16, 32, 16, 64, 16, 32, 16, 31),
    TADD_CASE("case_half_2x128_2x128_2x128_1x106", aclFloat16, 2, 128, 2, 128, 2, 128, 1, 106, true),
};
#endif

static void PrintUsage(const char *prog)
{
    std::cout << "Usage: " << prog << " [--all] [--case <name>]\n"
              << "  (default) run smoke-test cases only\n"
              << "  --all     run smoke + full regression cases (requires PTOISA_FULL_TEST build)\n"
              << "  --case    run a single named case\n";
}

static bool RunCases(const TAddCase *cases, size_t count, const std::string &dataRoot, const char *filterName)
{
    int failed = 0;
    int passed = 0;

    for (size_t i = 0; i < count; ++i) {
        if (filterName != nullptr && std::string(filterName) != cases[i].name) {
            continue;
        }

        const std::string dataDir = dataRoot + "/" + cases[i].name;
        INFO_LOG("Running %s", cases[i].name);
        if (cases[i].run(dataDir)) {
            INFO_LOG("PASS: %s", cases[i].name);
            passed++;
        } else {
            ERROR_LOG("FAIL: %s", cases[i].name);
            failed++;
        }
    }

    INFO_LOG("Summary: %d passed, %d failed", passed, failed);
    return failed == 0 && passed > 0;
}

int main(int argc, char **argv)
{
    bool runAll = false;
    const char *caseFilter = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--all") {
            runAll = true;
        } else if (std::string(argv[i]) == "--case" && i + 1 < argc) {
            caseFilter = argv[++i];
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else {
            ERROR_LOG("Unknown argument: %s", argv[i]);
            PrintUsage(argv[0]);
            return 1;
        }
    }

    const std::string dataRoot = "cases";
    bool ok = true;

    if (caseFilter != nullptr) {
        ok = RunCases(kSmokeCases, sizeof(kSmokeCases) / sizeof(kSmokeCases[0]), dataRoot, caseFilter);
#ifdef PTOISA_FULL_TEST
        if (!ok) {
            ok = RunCases(kFullCases, sizeof(kFullCases) / sizeof(kFullCases[0]), dataRoot, caseFilter);
        }
#else
        if (!ok) {
            ERROR_LOG("Case '%s' not found in smoke build. Rebuild with ./build.sh --full", caseFilter);
        }
#endif
        return ok ? 0 : 1;
    }

    ok = RunCases(kSmokeCases, sizeof(kSmokeCases) / sizeof(kSmokeCases[0]), dataRoot, nullptr);
    if (runAll) {
#ifdef PTOISA_FULL_TEST
        ok = RunCases(kFullCases, sizeof(kFullCases) / sizeof(kFullCases[0]), dataRoot, nullptr) && ok;
#else
        ERROR_LOG("--all requires a full build: ./build.sh --full");
        return 1;
#endif
    }

    return ok ? 0 : 1;
}
