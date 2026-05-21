/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
Pybind11 wrapper for ACL runtime used with runtime_camodel (simulator).
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "acl/acl.h"

namespace py = pybind11;

namespace {

void check_acl(aclError ret, const char *msg)
{
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ", ret=" + std::to_string(ret));
    }
}

} // namespace

PYBIND11_MODULE(pto_runtime, m)
{
    m.doc() = "ACL runtime bridge for PTO simulator tests (runtime_camodel)";

    m.def("init", []() { check_acl(aclInit(nullptr), "aclInit"); });

    m.def("finalize", []() { check_acl(aclFinalize(), "aclFinalize"); });

    m.def("set_device", [](int device_id) { check_acl(aclrtSetDevice(device_id), "aclrtSetDevice"); });

    m.def("reset_device", [](int device_id) { check_acl(aclrtResetDevice(device_id), "aclrtResetDevice"); });

    m.def("create_stream", []() -> uintptr_t {
        aclrtStream stream = nullptr;
        check_acl(aclrtCreateStream(&stream), "aclrtCreateStream");
        return reinterpret_cast<uintptr_t>(stream);
    });

    m.def("destroy_stream", [](uintptr_t stream) {
        check_acl(aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream)), "aclrtDestroyStream");
    });

    m.def("sync_stream", [](uintptr_t stream) {
        check_acl(aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream)), "aclrtSynchronizeStream");
    });

    m.def("malloc_host", [](size_t nbytes) -> uintptr_t {
        void *ptr = nullptr;
        check_acl(aclrtMallocHost(&ptr, nbytes), "aclrtMallocHost");
        return reinterpret_cast<uintptr_t>(ptr);
    });

    m.def("malloc_device", [](size_t nbytes) -> uintptr_t {
        void *ptr = nullptr;
        check_acl(aclrtMalloc(&ptr, nbytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        return reinterpret_cast<uintptr_t>(ptr);
    });

    m.def("free_host", [](uintptr_t ptr) {
        check_acl(aclrtFreeHost(reinterpret_cast<void *>(ptr)), "aclrtFreeHost");
    });

    m.def("free_device", [](uintptr_t ptr) {
        check_acl(aclrtFree(reinterpret_cast<void *>(ptr)), "aclrtFree");
    });

    m.def("memcpy_h2d", [](uintptr_t dst, uintptr_t src, size_t nbytes) {
        check_acl(
            aclrtMemcpy(reinterpret_cast<void *>(dst), nbytes, reinterpret_cast<void *>(src), nbytes,
                        ACL_MEMCPY_HOST_TO_DEVICE),
            "aclrtMemcpy H2D");
    });

    m.def("memcpy_d2h", [](uintptr_t dst, uintptr_t src, size_t nbytes) {
        check_acl(
            aclrtMemcpy(reinterpret_cast<void *>(dst), nbytes, reinterpret_cast<void *>(src), nbytes,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy D2H");
    });

    m.def("memcpy_h2d_bytes", [](uintptr_t dst, py::bytes src) {
        std::string data = src;
        check_acl(aclrtMemcpy(reinterpret_cast<void *>(dst), data.size(), data.data(), data.size(),
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy H2D bytes");
    });

    m.def("memcpy_d2h_bytes", [](uintptr_t src, size_t nbytes) -> py::bytes {
        std::string buf(nbytes, '\0');
        check_acl(aclrtMemcpy(buf.data(), nbytes, reinterpret_cast<void *>(src), nbytes, ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy D2H bytes");
        return py::bytes(buf);
    });
}
