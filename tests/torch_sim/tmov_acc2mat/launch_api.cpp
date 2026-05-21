#include <cstdint>

template <int32_t tilingKey>
void LaunchTMOVAcc2MatNZ2ND(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

extern "C" void pto_launch_nz2nd_4(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream)
{
    LaunchTMOVAcc2MatNZ2ND<4>(out, src0, src1, stream);
}
