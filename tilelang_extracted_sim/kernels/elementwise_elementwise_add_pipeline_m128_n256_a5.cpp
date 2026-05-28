#define PTO_PLATFORM_A5
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ float *A_handle, __gm__ float *B_handle, __gm__ float *C_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<float, 32, 128, 32, 128> a_ub;
  TASSIGN(a_ub, 0);
  tl::ascend_pto::TileUbDataND<float, 32, 128, 32, 128> b_ub;
  TASSIGN(b_ub, 16384);
  tl::ascend_pto::TileUbDataND<float, 32, 128, 32, 128> c_ub;
  TASSIGN(c_ub, 32768);
  auto vid = get_subblockid();
#if defined(__DAV_C310_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (0);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (1);
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (0);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 16, 128, 1, 1, 32768, 256, 1, 16, 128, pto::PadValue::Zero>(A_handle + ((vid * 4096) + (cid * 128)), 0, 0, 16, 128);
    tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 16, 128, 1, 1, 32768, 256, 1, 16, 128, pto::PadValue::Zero>(B_handle + ((vid * 4096) + (cid * 128)), 16384, 0, 16, 128);
    tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (0);

  for (int32_t mm = 0; mm < 4; ++mm) {
      if (mm < 3) {
        tl::ascend_pto::wait_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (((mm + 1) % 2));
        tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 16, 128, 1, 1, 32768, 256, 1, 16, 128, pto::PadValue::Zero>(A_handle + ((((mm * 8192) + (vid * 4096)) + (cid * 128)) + 8192), 0, (((mm + 1) % 2) * 2048), 16, 128);
        tl::ascend_pto::copy_gm_to_ub<float, float, 1, 1, 1, 16, 128, 1, 1, 32768, 256, 1, 16, 128, pto::PadValue::Zero>(B_handle + ((((mm * 8192) + (vid * 4096)) + (cid * 128)) + 8192), 16384, (((mm + 1) % 2) * 2048), 16, 128);
        tl::ascend_pto::set_flag_pipeline<PIPE_MTE2, PIPE_V> (((mm + 1) % 2));
      }
      tl::ascend_pto::wait_flag_pipeline<PIPE_MTE2, PIPE_V> ((mm % 2));
      tl::ascend_pto::TileUbDataND<float, 16, 128, 16, 128> a_ub_temp_0;
      TASSIGN(a_ub_temp_0, 0 + ((mm % 2) * 2048) * 4);
      tl::ascend_pto::TileUbDataND<float, 16, 128, 16, 128> b_ub_temp_0;
      TASSIGN(b_ub_temp_0, 16384 + ((mm % 2) * 2048) * 4);
      tl::ascend_pto::TileUbDataND<float, 16, 128, 16, 128> c_ub_temp_0;
      TASSIGN(c_ub_temp_0, 32768 + ((mm % 2) * 2048) * 4);
      TADD(c_ub_temp_0, a_ub_temp_0, b_ub_temp_0);
      tl::ascend_pto::set_flag_pipeline<PIPE_V, PIPE_MTE3> ((mm % 2));
      tl::ascend_pto::wait_flag_pipeline<PIPE_V, PIPE_MTE3> ((mm % 2));
      tl::ascend_pto::copy_ub_to_gm<float, float, 1, 1, 1, 16, 128, 1, 1, 32768, 256, 1, 16, 128>(C_handle + (((mm * 8192) + (vid * 4096)) + (cid * 128)), 32768, ((mm % 2) * 2048), 16, 128);
      tl::ascend_pto::set_flag_pipeline<PIPE_MTE3, PIPE_MTE2> ((mm % 2));
    }
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (0);
    tl::ascend_pto::wait_flag_pipeline<PIPE_MTE3, PIPE_MTE2> (1);
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
    launch_kernel<<<2, nullptr, stream>>>(A_handle, B_handle, C_handle, fftsAddr);
}
