#include <cstdint>

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW,
          int vRows, int vCols>
void LaunchTAdd(T *out, T *src0, T *src1, void *stream);

extern "C" void pto_tadd_float_64x64(float *out, float *src0, float *src1, void *stream)
{
    LaunchTAdd<float, 64, 64, 64, 64, 64, 64, 64, 64>(out, src0, src1, stream);
}

extern "C" void pto_tadd_int32_64x64(int32_t *out, int32_t *src0, int32_t *src1, void *stream)
{
    LaunchTAdd<int32_t, 64, 64, 64, 64, 64, 64, 64, 64>(out, src0, src1, stream);
}
