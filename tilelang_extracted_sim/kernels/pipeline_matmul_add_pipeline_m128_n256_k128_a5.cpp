#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *A_handle, __gm__ half *B_handle, __gm__ half *C_handle, __gm__ half *D_handle, __gm__ half *workspace_1_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileMatL1<half, 384, 64, 384, 64> A_L1;
  TASSIGN(A_L1, 0);
  tl::ascend_pto::TileMatL1<half, 192, 256, 192, 256> B_L1;
  TASSIGN(B_L1, 49152);
  TileAcc<float, 128, 256, 128, 256> C_L0;
  TASSIGN(C_L0, 0);
  tl::ascend_pto::TileUbDataND<half, 128, 64, 128, 64> c_ub;
  TASSIGN(c_ub, 0);
  tl::ascend_pto::TileUbDataND<half, 128, 64, 128, 64> d_ub;
  TASSIGN(d_ub, 16384);
  tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> e_ub;
  TASSIGN(e_ub, 32768);
  auto vid = get_subblockid();
#if defined(__DAV_C310_CUBE__)
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 64, 1, 1, 16384, 128, 1, 128, 64>(A_handle + 0, 0, 0, 128, 64);
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256>(B_handle + 0, 49152, 0, 64, 256);
    pipe_barrier(PIPE_MTE2);
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 128, 64, 1, 1, 16384, 128, 1, 128, 64>(A_handle + 64, 0, 8192, 128, 64);
    tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 256, 1, 1, 32768, 256, 1, 64, 256>(B_handle + 16384, 49152, 16384, 64, 256);
    set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
    tl::ascend_pto::TileMatL1<half, 128, 64, 128, 64> A_L1_temp_0;
    TASSIGN(A_L1_temp_0, 0 + 0 * 2);
    tl::ascend_pto::TileMatL1<half, 64, 256, 64, 256> B_L1_temp_0;
    TASSIGN(B_L1_temp_0, 49152 + 0 * 2);
    tl::ascend_pto::gemm_v0<half, float, 128, 256, 64, 128, 256, 64, 64, false, false>(A_L1_temp_0, B_L1_temp_0, C_L0, (bool)1);
    pipe_barrier(PIPE_M);
    tl::ascend_pto::TileMatL1<half, 128, 64, 128, 64> A_L1_temp_1;
    TASSIGN(A_L1_temp_1, 0 + 8192 * 2);
    tl::ascend_pto::TileMatL1<half, 64, 256, 64, 256> B_L1_temp_1;
    TASSIGN(B_L1_temp_1, 49152 + 16384 * 2);
    tl::ascend_pto::gemm_v0<half, float, 128, 256, 64, 128, 256, 64, 64, false, false>(A_L1_temp_1, B_L1_temp_1, C_L0, (bool)0);
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID2);
    tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 128, 256, 1, 1, 32768, 256, 1, 128, 256>(workspace_1_handle + 0, 0, 0, 128, 256);
    set_intra_block(PIPE_FIX, 0);
    set_intra_block(PIPE_FIX, 16);
#endif
#if defined(__DAV_C310_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::wait_intra_block_vec<PIPE_MTE2>(0);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64, pto::PadValue::Zero>(workspace_1_handle + (vid * 16384), 0, 0, 64, 64);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64, pto::PadValue::Zero>(D_handle + (vid * 16384), 16384, 0, 64, 64);
    pipe_barrier(PIPE_MTE2);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64, pto::PadValue::Zero>(workspace_1_handle + ((vid * 16384) + 64), 0, 4096, 64, 64);
    tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64, pto::PadValue::Zero>(D_handle + ((vid * 16384) + 64), 16384, 4096, 64, 64);

  for (int32_t i = 0; i < 2; ++i) {
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID7);
      tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> c_ub_temp_0;
      TASSIGN(c_ub_temp_0, 0 + (i * 4096) * 2);
      tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> d_ub_temp_0;
      TASSIGN(d_ub_temp_0, 16384 + (i * 4096) * 2);
      tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> e_ub_temp_0;
      TASSIGN(e_ub_temp_0, 32768 + 0 * 2);
      TADD(e_ub_temp_0, c_ub_temp_0, d_ub_temp_0);
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64, pto::PadValue::Zero>(workspace_1_handle + (((vid * 16384) + (i * 64)) + 128), 0, (i * 4096), 64, 64);
      tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64, pto::PadValue::Zero>(D_handle + (((vid * 16384) + (i * 64)) + 128), 16384, (i * 4096), 64, 64);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID5);
      pipe_barrier(PIPE_MTE3);
      tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64>(C_handle + ((vid * 16384) + (i * 64)), 32768, 0, 64, 64);
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
    tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> c_ub_temp_1;
    TASSIGN(c_ub_temp_1, 0 + 0 * 2);
    tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> d_ub_temp_1;
    TASSIGN(d_ub_temp_1, 16384 + 0 * 2);
    tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> e_ub_temp_1;
    TASSIGN(e_ub_temp_1, 32768 + 0 * 2);
    TADD(e_ub_temp_1, c_ub_temp_1, d_ub_temp_1);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID4);
    pipe_barrier(PIPE_MTE3);
    tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64>(C_handle + ((vid * 16384) + 128), 32768, 0, 64, 64);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> c_ub_temp_2;
    TASSIGN(c_ub_temp_2, 0 + 4096 * 2);
    tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> d_ub_temp_2;
    TASSIGN(d_ub_temp_2, 16384 + 4096 * 2);
    tl::ascend_pto::TileUbDataND<half, 64, 64, 64, 64> e_ub_temp_2;
    TASSIGN(e_ub_temp_2, 32768 + 0 * 2);
    TADD(e_ub_temp_2, c_ub_temp_2, d_ub_temp_2);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
    pipe_barrier(PIPE_MTE3);
    tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 64, 1, 1, 32768, 256, 1, 64, 64>(C_handle + ((vid * 16384) + 192), 32768, 0, 64, 64);
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
