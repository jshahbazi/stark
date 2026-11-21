"""Build preparation and reproducible PNG controls for pinned vision parity."""

import shutil
import struct
import sys
import zlib
from pathlib import Path


def instrument(source):
    runner = source / "mlxrunner"
    shutil.copyfile(Path(__file__).with_name("capture.go"), runner / "inkling_vision_capture.go")
    server = runner / "server.go"
    original = "defer runner.Close()"
    data = server.read_text()
    if data.count(original) != 1:
        raise ValueError("unexpected pinned server entry point")
    server.write_text(data.replace(
        original,
        original + """
    if images := os.Getenv("INKLING_VISION_IMAGES"); images != "" {
        return worker.Do(runnerCtx, func() error {
            return captureInklingVision(runner.Model, images, os.Getenv("INKLING_VISION_OUTPUT"))
        })
    }""",
    ))
    directory = source / "cmd" / "inkling-vision-oracle"
    directory.mkdir(exist_ok=True)
    (directory / "main.go").write_text('''package main

import (
    "log"
    "os"
    "github.com/ollama/ollama/mlxrunner"
)

func main() {
    if err := mlxrunner.Execute(os.Args[1:]); err != nil {
        log.Fatal(err)
    }
}
''')


def png(path, width, height, transparent):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    def pixels(x, y):
        rgb = [(x * 23 + y * 17) % 256, (x * 2 + y * 23) % 256, (x * 7 + y * 11) % 256]
        return bytes(rgb + ([(x * 3 + y * 9) % 256] if transparent else []))

    rows = b"".join(b"\0" + b"".join(pixels(x, y) for x in range(width)) for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6 if transparent else 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    instrument(Path(sys.argv[1]))
    fixtures = Path(sys.argv[2])
    fixtures.mkdir(parents=True, exist_ok=True)
    png(fixtures / "opaque.png", 40, 40, False)
    png(fixtures / "padded-alpha.png", 84, 53, True)
