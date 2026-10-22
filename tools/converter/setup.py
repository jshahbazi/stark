"""Ship checkout inputs under their logical paths for installed provenance."""

import shutil
from pathlib import Path
from runpy import run_path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
SOURCE_INPUTS = run_path(str(ROOT / "stark_convert" / "_build_metadata.py"))["SOURCE_INPUTS"]


class ProvenanceBuild(build_py):
    def run(self):
        super().run()
        if self.editable_mode:
            return
        capsule = Path(self.build_lib) / "stark_convert" / "_build_inputs"
        if capsule.exists():
            shutil.rmtree(capsule)
        for name in (*SOURCE_INPUTS, "uv.lock"):
            destination = capsule / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)

    def get_source_files(self):
        return super().get_source_files() + [*SOURCE_INPUTS, "uv.lock"]

    def get_outputs(self, include_bytecode=1):
        outputs = super().get_outputs(include_bytecode)
        if not self.editable_mode:
            outputs.extend(
                str(Path(self.build_lib) / "stark_convert" / "_build_inputs" / name)
                for name in (*SOURCE_INPUTS, "uv.lock")
            )
        return outputs


setup(cmdclass={"build_py": ProvenanceBuild})
