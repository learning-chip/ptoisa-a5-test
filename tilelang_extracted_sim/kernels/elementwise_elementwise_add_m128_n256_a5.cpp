#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ float *A_handle, __gm__ float *B_handle, __gm__ float *C_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<float, 64, 256, 64, 256> a_ub;
  TASSIGN(a_ub, 0);
  tl::ascend_pto::TileUbDataND<float, 64, 256, 64, 256> b_ub;
  TASSIGN(b_ub, 65536);
  tl::ascend_pto::TileUbDataND<float, 64, 256, 64, 256> c_ub;
  TASSIGN(c_ub, 131072);
  auto vid = get_subblockid();
#if defined(__DAV_C310_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256, pto::PadValue::Zero>(A_handle + (vid * 16384), 0, 0, 64, 256);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256, pto::PadValue::Zero>(B_handle + (vid * 16384), 65536, 0, 64, 256);
    pipe_barrier(PIPE_ALL);
    TADD(c_ub, a_ub, b_ub);
    pipe_barrier(PIPE_ALL);
    tl::ascend_pto::copy_ub_to_gm<float, float, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256>(C_handle + (vid * 16384), 131072, 0, 64, 256);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *A_handle, __gm__ uint8_t *B_handle, __gm__ uint8_t *C_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ float *>(A_handle),
     reinterpret_cast<__gm__ float *>(B_handle),
     reinterpret_cast<__gm__ float *>(C_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *A_handle, uint8_t *B_handle, uint8_t *C_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<1, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
