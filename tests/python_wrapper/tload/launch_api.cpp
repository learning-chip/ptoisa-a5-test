#include <cstdint>

template <int32_t testKey>
void launchTLOAD(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream);

template <int32_t testKey>
int get_input_golden(uint8_t *input, uint8_t *golden);

extern "C" int pto_get_input_golden_1(uint8_t *input, uint8_t *golden)
{
    return get_input_golden<1>(input, golden);
}

extern "C" void pto_launch_tload_1(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream)
{
    launchTLOAD<1>(out, src, gLog, stream);
}
