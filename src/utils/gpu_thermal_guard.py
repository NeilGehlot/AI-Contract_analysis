"""Pause GPU training when a laptop runs hot. Does not kill the CUDA context mid-kernel."""

import subprocess
import time

from transformers import TrainerCallback


def gpu_temperature_c() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class GpuThermalGuard(TrainerCallback):
    """Sleep between steps if the GPU is too hot. Abort only at a hard limit."""

    def __init__(
        self,
        pause_c: int = 90,
        resume_c: int = 84,
        abort_c: int = 95,
        check_every: int = 1,
    ):
        self.pause_c = pause_c
        self.resume_c = resume_c
        self.abort_c = abort_c
        self.check_every = check_every
        print(
            f"GPU thermal guard: pause at {pause_c}C, resume at {resume_c}C, stop at {abort_c}C",
            flush=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_every != 0:
            return
        temp = gpu_temperature_c()
        if temp is None:
            return
        if temp >= self.abort_c:
            raise RuntimeError(
                f"GPU temperature {temp}C reached the {self.abort_c}C stop limit."
            )
        while temp is not None and temp >= self.pause_c:
            print(
                f"GPU {temp}C >= {self.pause_c}C; pausing until it drops to {self.resume_c}C.",
                flush=True,
            )
            time.sleep(20)
            temp = gpu_temperature_c()
            if temp is not None and temp >= self.abort_c:
                raise RuntimeError(
                    f"GPU temperature {temp}C reached the {self.abort_c}C stop limit."
                )
