/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "acl/acl.h"
#include <pto/common/type.hpp>

namespace PtoTestCommon {

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

inline bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    if (stat(filePath.data(), &sBuf) == -1) {
        ERROR_LOG("Failed to get file. Path = %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. Path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("%s: file size (%lu) is larger than buffer size (%lu)", filePath.c_str(), size, bufferSize);
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    std::ofstream file(filePath, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }
    file.write(static_cast<const char *>(buffer), size);
    return file.good();
}

template <typename T>
bool ResultCmp(const std::vector<T> &outDataValExp, const std::vector<T> &outDataValAct, float eps,
               size_t threshold = 0, size_t zeroCountThreshold = 1000)
{
    if (outDataValExp.size() != outDataValAct.size()) {
        std::cout << "out size is not eq, golden: " << outDataValExp.size() << ", act: " << outDataValAct.size()
                  << std::endl;
        return false;
    }

    threshold = threshold == 0 ? static_cast<size_t>(outDataValExp.size() * eps) : threshold;

    size_t errCount = 0;
    size_t zeroCount = 0;
    float maxDiff = 0;

    for (size_t eIdx = 0; eIdx < outDataValExp.size(); eIdx++) {
        auto expVal = static_cast<float>(outDataValExp[eIdx]);
        auto actVal = static_cast<float>(outDataValAct[eIdx]);
        auto diff = std::abs(expVal - actVal);
        auto relRatio = expVal == 0.0f ? diff : std::abs(diff / expVal);
        maxDiff = std::max(diff, maxDiff);
        zeroCount += std::abs(actVal) <= 1e-6f && std::abs(expVal) > 1e-6f ? 1 : 0;
        if ((diff > eps && relRatio > eps) || zeroCount > zeroCountThreshold) {
            errCount++;
        }
    }

    bool ok = errCount <= threshold && zeroCount <= zeroCountThreshold;
    std::cout << "max diff: " << maxDiff << ", err count: " << errCount << ", err threshold: " << threshold
              << std::endl;
    return ok;
}

} // namespace PtoTestCommon
