"""Watch the CPU thermal zone during GPU training.

Training is not moved to the CPU. If the zone is too hot, the trainer sleeps
or stops. The Windows counter is kelvin, so values above 200 are converted
with 273.15 subtracted.
"""

import subprocess
import threading
import time

from transformers import TrainerCallback

_READER_SCRIPT = r"""
while ($true) {
  try {
    $samples = (Get-Counter '\Thermal Zone Information(*)\Temperature' -ErrorAction Stop).CounterSamples
    $max = ($samples | Measure-Object -Property CookedValue -Maximum).Maximum
    Write-Output $max
  } catch {
    Write-Output 'NA'
  }
  Start-Sleep -Seconds 10
}
"""


def celsius_from_counter(value: float) -> int | None:
    if value > 200:
        return int(round(value - 273.15))
    if 0 < value < 150:
        return int(round(value))
    return None


class CpuTempReader:
    """Background reader so a counter query does not stall each training step."""

    def __init__(self) -> None:
        self.temp_c: int | None = None
        self._proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", _READER_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        if self._proc.stdout is None:
            return
        for line in self._proc.stdout:
            text = line.strip()
            if not text or text == "NA":
                continue
            try:
                self.temp_c = celsius_from_counter(float(text))
            except ValueError:
                continue

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.kill()


class CpuThermalGuard(TrainerCallback):
    """Pause or stop when the CPU thermal zone is too hot. Device stays CUDA."""

    def __init__(
        self,
        pause_c: int = 97,
        resume_c: int = 92,
        abort_c: int = 100,
        check_every: int = 5,
    ):
        self.pause_c = pause_c
        self.resume_c = resume_c
        self.abort_c = abort_c
        self.check_every = check_every
        self.reader = CpuTempReader()
        self._announced = False
        print(
            f"CPU thermal guard: pause at {pause_c}C, resume at {resume_c}C, stop at {abort_c}C. "
            "Training stays on the GPU.",
            flush=True,
        )

    def _temp(self) -> int | None:
        temp = self.reader.temp_c
        if temp is not None and not self._announced:
            print(f"CPU thermal zone {temp}C.", flush=True)
            self._announced = True
        return temp

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_every != 0:
            return
        temp = self._temp()
        if temp is None:
            return
        if temp >= self.abort_c:
            self.reader.close()
            raise RuntimeError(
                f"CPU thermal zone {temp}C reached the {self.abort_c}C stop limit. Training stays on the GPU."
            )
        while temp is not None and temp >= self.pause_c:
            print(
                f"CPU thermal zone {temp}C >= {self.pause_c}C; pausing until it drops to {self.resume_c}C. GPU training only.",
                flush=True,
            )
            time.sleep(20)
            temp = self._temp()
            if temp is not None and temp >= self.abort_c:
                self.reader.close()
                raise RuntimeError(
                    f"CPU thermal zone {temp}C reached the {self.abort_c}C stop limit. Training stays on the GPU."
                )

    def on_train_end(self, args, state, control, **kwargs):
        self.reader.close()
