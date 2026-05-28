#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ int8_t *A_handle, __gm__ int8_t *B_handle, __gm__ float *scale_handle, __gm__ bfloat16_t *C_handle, __gm__ int *workspace_1_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileMatL1<int8_t, 128, 64, 128, 64> A_L1;
  TASSIGN(A_L1, 0);
  tl::ascend_pto::TileMatL1<int8_t, 64, 256, 64, 256> B_L1;
  TASSIGN(B_L1, 8192);
  TileAcc<int, 128, 256, 128, 256> C_L0;
  TASSIGN(C_L0, 0);
  tl::ascend_pto::TileUbDataND<int, 64, 256, 64, 256> c_ub;
  TASSIGN(c_ub, 0);
  tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> scale_ub;
  TASSIGN(scale_ub, 65536);
  tl::ascend_pto::TileUbDataND<float, 64, 256, 64, 256> c_scale;
  TASSIGN(c_scale, 66560);
  tl::ascend_pto::TileUbDataND<bfloat16_t, 64, 256, 64, 256> c_out;
  TASSIGN(c_out, 132096);
  auto vid = get_subblockid();
#if defined(__DAV_C310_CUBE__)
    tl::ascend_pto::copy_gm_to_l1<int8_t, int8_t, 1, 1, 1, 128, 64, 1, 1, 4096, 64, 1, 128, 64>(A_handle + 0, 0, 0, 64, 64);
    tl::ascend_pto::copy_gm_to_l1<int8_t, int8_t, 1, 1, 1, 64, 256, 1, 1, 4096, 64, 1, 64, 256>(B_handle + 0, 8192, 0, 64, 64);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
    tl::ascend_pto::gemm_v0<int8_t, int, 128, 256, 64, 128, 256, 64, 64, false, false>(A_L1, B_L1, C_L0, (bool)1);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    tl::ascend_pto::copy_l0c_to_gm<int, int, 1, 1, 1, 128, 256, 1, 1, 4096, 64, 1, 128, 256>(workspace_1_handle + 0, 0, 0, 64, 64);
    set_intra_block(PIPE_FIX, 0);
    set_intra_block(PIPE_FIX, 16);
#endif
#if defined(__DAV_C310_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::wait_intra_block_vec<PIPE_MTE2>(0);
    tl::ascend_pto::copy_gm_to_ub<int, int, 1, 1, 1, 64, 256, 1, 1, 4096, 64, 1, 64, 256, pto::PadValue::Zero>(workspace_1_handle + (vid * 4096), 0, 0, ((vid == 0) ? 64 : 0), 64);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 1, 256, 1, 1, 1, 1, 1, 1, 256, pto::PadValue::Zero>(scale_handle + 0, 65536, 0, 1, 1);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(scale_ub, scale_ub.GetValue(0));
TCVT(c_scale, c_ub, RoundMode::CAST_RINT);

  for (int32_t outer_broadcast_idx = 0; outer_broadcast_idx < 64; ++outer_broadcast_idx) {
      tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> c_scale_temp_0;
      TASSIGN(c_scale_temp_0, 66560 + (outer_broadcast_idx * 256) * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> scale_ub_temp_0;
      TASSIGN(scale_ub_temp_0, 65536 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 256, 1, 256> c_scale_temp_1;
      TASSIGN(c_scale_temp_1, 66560 + (outer_broadcast_idx * 256) * 4);
      TMUL(c_scale_temp_1, c_scale_temp_0, scale_ub_temp_0);
    }
TCVT(c_out, c_scale, RoundMode::CAST_RINT);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    tl::ascend_pto::copy_ub_to_gm<bfloat16_t, bfloat16_t, 1, 1, 1, 64, 256, 1, 1, 4096, 64, 1, 64, 256>(C_handle + (vid * 4096), 132096, 0, ((vid == 0) ? 64 : 0), 64);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *A_handle, __gm__ uint8_t *B_handle, __gm__ uint8_t *scale_handle, __gm__ uint8_t *C_handle, __gm__ uint8_t *workspace_1_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ int8_t *>(A_handle),
     reinterpret_cast<__gm__ int8_t *>(B_handle),
     reinterpret_cast<__gm__ float *>(scale_handle),
     reinterpret_cast<__gm__ bfloat16_t *>(C_handle),
     reinterpret_cast<__gm__ int *>(workspace_1_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *A_handle, uint8_t *B_handle, uint8_t *scale_handle, uint8_t *C_handle, uint8_t *workspace_1_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<1, nullptr, stream>>>(A_handle, B_handle, scale_handle, C_handle, workspace_1_handle, fftsAddr);
}
