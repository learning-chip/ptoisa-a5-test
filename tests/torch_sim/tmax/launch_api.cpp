#include <cstdint>

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW,
          int vRows, int vCols, bool sameTile>
void LaunchTMax(T *out, T *src0, T *src1, void *stream);

extern "C" void pto_tmax_float_16x32(float *out, float *src0, float *src1, void *stream)
{
    LaunchTMax<float, 16, 32, 16, 64, 16, 32, 16, 32, false>(out, src0, src1, stream);
}

extern "C" void pto_tmax_int32_16x32(int32_t *out, int32_t *src0, int32_t *src1, void *stream)
{
    LaunchTMax<int32_t, 16, 32, 16, 64, 16, 32, 16, 32, false>(out, src0, src1, stream);
}
