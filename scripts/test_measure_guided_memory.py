"""Exercise the measurement callback through the production telemetry reader."""

import os
import time

from measure_guided_memory import Monitor
from stark_convert.chat import Worker


def test_worker_telemetry_reaches_measurement_monitor():
    monitor = Monitor()
    worker = Worker.__new__(Worker)
    worker._closed = False
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    worker._start_gpu_memory_reader(reader, monitor.gpu)
    try:
        os.write(writer, b'{"gpu_memory_bytes":4096}\n')
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with monitor.lock:
                if (
                    monitor.phases.get("load", {}).get("peak_metal_allocated_bytes")
                    == 4096
                ):
                    break
            time.sleep(0.01)
        with monitor.lock:
            assert monitor.phases["load"]["peak_metal_allocated_bytes"] == 4096
    finally:
        worker._close_gpu_memory_reader()
        os.close(writer)
