#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *A_handle, __gm__ half *B_handle, __gm__ half *C_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  TileAcc<float, 64, 64, 64, 64> C_L0;
  TASSIGN(C_L0, 0);
  tl::ascend_pto::TileMatL1<half, 64, 64, 64, 64> A_L1;
  TASSIGN(A_L1, 0);
  tl::ascend_pto::TileMatL1<half, 64, 64, 64, 64> B_L1;
  TASSIGN(B_L1, 8192);
#if defined(__DAV_C310_CUBE__)

  for (int32_t k = 0; k < 9; ++k) {
      set_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID2);
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 64, 1, 1, 300223, 539, 1, 64, 64>(A_handle + (((cid / 8) * 34496) + (k * 64)), 0, 0, ((cid <= 63) ? 64 : (557 - ((cid / 8) * 64))), ((k <= 7) ? 64 : (539 - (k * 64))));
      tl::ascend_pto::copy_gm_to_l1<half, half, 1, 1, 1, 64, 64, 1, 1, 275968, 512, 1, 64, 64>(B_handle + ((k * 32768) + ((cid % 8) * 64)), 8192, 0, ((k <= 7) ? 64 : (539 - (k * 64))), 64);
      set_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_M, EVENT_ID1);
      pipe_barrier(PIPE_M);
      tl::ascend_pto::gemm_v0<half, float, 64, 64, 64, 64, 64, 64, 64, false, false>(A_L1, B_L1, C_L0, (k == 0));
    }
    set_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID4);
    tl::ascend_pto::copy_l0c_to_gm<half, float, 1, 1, 1, 64, 64, 1, 1, 285184, 512, 1, 64, 64>(C_handle + (((cid / 8) * 32768) + ((cid % 8) * 64)), 0, 0, ((cid <= 63) ? 64 : (557 - ((cid / 8) * 64))), 64);
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
    launch_kernel<<<72, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
