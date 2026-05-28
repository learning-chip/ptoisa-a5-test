#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *A_handle, __gm__ half *B_handle, __gm__ half *C_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  TileAcc<float, 32, 32, 32, 32> C_L0;
  TASSIGN(C_L0, 0);
  tl::ascend_pto::TileMatL1<half, 32, 32, 32, 32> A_L1;
  TASSIGN(A_L1, 0);
  tl::ascend_pto::TileMatL1<half, 32, 32, 32, 32> B_L1;
  TASSIGN(B_L1, 2048);
#if defined(__DAV_C310_CUBE__)

  for (int32_t k = 0; k < 5; ++k) {
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 32, 32, 1, 1, 20034, 159, 1, 32, 32>(A_handle + (((cid / 3) * 5088) + (k * 32)), 0, 0, ((cid <= 8) ? 32 : (126 - ((cid / 3) * 32))), ((k <= 3) ? 32 : (159 - (k * 32))));
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 32, 32, 1, 1, 12720, 80, 1, 32, 32>(B_handle + ((k * 2560) + ((cid % 3) * 32)), 2048, 0, ((k <= 3) ? 32 : (159 - (k * 32))), (((cid % 3) <= 1) ? 32 : (80 - ((cid % 3) * 32))));
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      pipe_barrier(PIPE_M);
      tl::ascend_pto::gemm_v0<half, float, 32, 32, 32, 32, 32, 32, 32, false, false>(A_L1, B_L1, C_L0, (k == 0));
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
    tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 32, 32, 1, 1, 10080, 80, 1, 32, 32>(C_handle + (((cid / 3) * 2560) + ((cid % 3) * 32)), 0, 0, ((cid <= 8) ? 32 : (126 - ((cid / 3) * 32))), (((cid % 3) <= 1) ? 32 : (80 - ((cid % 3) * 32))));
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *A_handle, __gm__ uint8_t *B_handle, __gm__ uint8_t *C_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(A_handle),
     reinterpret_cast<__gm__ half *>(B_handle),
     reinterpret_cast<__gm__ half *>(C_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *A_handle, uint8_t *B_handle, uint8_t *C_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<12, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
