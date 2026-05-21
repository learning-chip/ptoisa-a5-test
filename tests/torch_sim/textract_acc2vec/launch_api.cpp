#include <cstdint>

template <int32_t tilingKey>
void LaunchTMOVAcc2VecNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

extern "C" void pto_launch_nz2nd_3(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream)
{
    LaunchTMOVAcc2VecNZ2ND<3>(out, src0, src1, src2, stream);
}
