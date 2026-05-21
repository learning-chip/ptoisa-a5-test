"""Configure LD_LIBRARY_PATH for CANN simulator (camodel) execution."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

_ENV_FLAG = "_PTO_PYTHON_SIM_READY"


def _preload_shared_libs(soc_version: str) -> None:
    """Load simulator DSOs by absolute path (os.environ LD changes are too late for pybind import)."""
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        return
    ascend = Path(ascend_home)
    sim_lib = ascend / "tools" / "simulator" / soc_version / "lib"
    mode = getattr(ctypes, "RTLD_GLOBAL", 0x100)
    for lib in sorted(sim_lib.glob("*.so")):
        try:
            ctypes.CDLL(str(lib), mode=mode)
        except OSError:
            pass
    ascend_cl = ascend / "lib64" / "libascendcl.so"
    if ascend_cl.is_file():
        ctypes.CDLL(str(ascend_cl), mode=mode)


def setup_simulator_env(soc_version: str | None = None, reexec: bool = True) -> None:
    """Mirror tests/cpp/common/sim_env.sh before loading native extensions."""
    soc = soc_version or os.environ.get("SOC_VERSION", "Ascend950PR_9599")
    if os.environ.get(_ENV_FLAG) == "1":
        sim_lib = f"{os.environ['ASCEND_HOME_PATH']}/tools/simulator/{soc}/lib"
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if sim_lib not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{sim_lib}:{ld}" if ld else sim_lib
        return

    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        raise EnvironmentError("ASCEND_HOME_PATH is not set. Source CANN setenv.bash first.")

    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ld:
        parts = [p for p in ld.split(":") if p and "/runtime/lib64" not in p]
        ld = ":".join(parts)

    stub = f"{ascend_home}/runtime/lib64/stub"
    os.environ["LD_LIBRARY_PATH"] = f"{stub}:{ld}" if ld else stub

    setenv = Path(ascend_home) / "bin" / "setenv.bash"
    if setenv.is_file():
        result = subprocess.run(
            f"source {setenv} && env",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                os.environ[key] = value

    sim_lib = f"{ascend_home}/tools/simulator/{soc}/lib"
    os.environ["LD_LIBRARY_PATH"] = f"{sim_lib}:{os.environ['LD_LIBRARY_PATH']}"
    os.environ[_ENV_FLAG] = "1"
    if reexec:
        os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)
