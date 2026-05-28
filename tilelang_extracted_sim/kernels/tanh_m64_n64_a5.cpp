#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ float *A_handle, __gm__ float *B_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<float, 32, 64, 32, 64> a_ub;
  TASSIGN(a_ub, 0);
  tl::ascend_pto::TileUbDataND<float, 32, 64, 32, 64> zero_ub;
  TASSIGN(zero_ub, 8192);
  tl::ascend_pto::TileUbDataND<float, 32, 64, 32, 64> nega_ub;
  TASSIGN(nega_ub, 16384);
  tl::ascend_pto::TileUbDataND<float, 32, 64, 32, 64> b_ub;
  TASSIGN(b_ub, 24576);
  tl::ascend_pto::TileUbDataND<float, 32, 64, 32, 64> denom_ub;
  TASSIGN(denom_ub, 32768);
  auto vid = get_subblockid();
#if defined(__DAV_C310_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 32, 64, 1, 1, 4096, 64, 1, 32, 64, pto::PadValue::Zero>(A_handle + (vid * 2048), 0, 0, 32, 64);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.000000e+00f);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TSUB(nega_ub, zero_ub, a_ub);
    TEXP(a_ub, a_ub);
    TEXP(nega_ub, nega_ub);
    TSUB(b_ub, a_ub, nega_ub);
    TADD(denom_ub, a_ub, nega_ub);
    TDIV(b_ub, b_ub, denom_ub);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    tl::ascend_pto::copy_ub_to_gm<float, float, 1, 1, 1, 32, 64, 1, 1, 4096, 64, 1, 32, 64>(B_handle + (vid * 2048), 24576, 0, 32, 64);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *A_handle, __gm__ uint8_t *B_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ float *>(A_handle),
     reinterpret_cast<__gm__ float *>(B_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *A_handle, uint8_t *B_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<1, nullptr, stream>>>(A_handle, B_handle, fftsAddr);
}
