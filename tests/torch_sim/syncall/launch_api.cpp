#include <cstdint>

void LaunchHardSyncAll(int32_t *out, int32_t *flags, int32_t totalBlocks, void *stream);

extern "C" void pto_launch_hard_syncall_18(int32_t *out, int32_t *flags, void *stream)
{
    LaunchHardSyncAll(out, flags, 18, stream);
}
