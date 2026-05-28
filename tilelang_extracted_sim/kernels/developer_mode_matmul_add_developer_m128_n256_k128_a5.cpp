#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *A_handle, __gm__ half *B_handle, __gm__ half *C_handle, __gm__ half *D_handle, __gm__ half *workspace_1_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  TileAcc<float, 128, 256, 128, 256> C_L0;
  TASSIGN(C_L0, 0);
  tl::ascend_pto::TileMatL1<half, 128, 64, 128, 64> A_L1;
  TASSIGN(A_L1, 0);
  tl::ascend_pto::TileMatL1<half, 64, 256, 64, 256> B_L1;
  TASSIGN(B_L1, 16384);
  tl::ascend_pto::TileUbDataND<half, 64, 256, 64, 256> d_ub;
  TASSIGN(d_ub, 0);
  tl::ascend_pto::TileUbDataND<half, 64, 256, 64, 256> c_ub;
  TASSIGN(c_ub, 32768);
  auto vid = get_subblockid();
#if defined(__DAV_C310_CUBE__)

  for (int32_t k = 0; k < 2; ++k) {
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 64, 1, 1, 16384, 128, 1, 128, 64>(A_handle + (k * 64), 0, 0, 128, 64);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256>(B_handle + (k * 16384), 16384, 0, 64, 256);
      pipe_barrier(PIPE_ALL);
      if (k == 0) {
        tl::ascend_pto::gemm_v0<half, float, 128, 256, 64, 128, 256, 64, 64, false, false>(A_L1, B_L1, C_L0, (bool)1);
      } else {
        tl::ascend_pto::gemm_v0<half, float, 128, 256, 64, 128, 256, 64, 64, false, false>(A_L1, B_L1, C_L0, (bool)0);
      }
      pipe_barrier(PIPE_ALL);
      pipe_barrier(PIPE_ALL);
    }
    tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 256, 1, 32768, 32768, 256, 1, 128, 256>(workspace_1_handle + 0, 0, 0, 0, 0);
    set_intra_block(PIPE_FIX, 0);
    set_intra_block(PIPE_FIX, 16);
#endif
#if defined(__DAV_C310_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256, pto::PadValue::Zero>(D_handle + (vid * 16384), 0, 0, 64, 256);
    tl::ascend_pto::wait_intra_block_vec<PIPE_MTE2>(0);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 256, 1, 32768, 32768, 256, 1, 64, 256, pto::PadValue::Zero>(workspace_1_handle + (vid * 16384), 32768, 0, 64, 256);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TADD(c_ub, c_ub, d_ub);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256>(C_handle + (vid * 16384), 32768, 0, 64, 256);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *A_handle, __gm__ uint8_t *B_handle, __gm__ uint8_t *C_handle, __gm__ uint8_t *D_handle, __gm__ uint8_t *workspace_1_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(A_handle),
     reinterpret_cast<__gm__ half *>(B_handle),
     reinterpret_cast<__gm__ half *>(C_handle),
     reinterpret_cast<__gm__ half *>(D_handle),
     reinterpret_cast<__gm__ half *>(workspace_1_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *A_handle, uint8_t *B_handle, uint8_t *C_handle, uint8_t *D_handle, uint8_t *workspace_1_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<1, nullptr, stream>>>(A_handle, B_handle, C_handle, D_handle, workspace_1_handle, fftsAddr);
}
