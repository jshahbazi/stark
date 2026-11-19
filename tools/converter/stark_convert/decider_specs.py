"""Immutable Decider v1.3 source profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeciderSourceProfile:
    family_id: str
    name: str
    repo: str
    revision: str
    base_model: str
    tied_word_embeddings: bool
    parameter_count: int
    shard_count: int
    size_label: str
    file_pins: dict[str, tuple[str, str, int]]

    @property
    def weight_bytes(self) -> int:
        return sum(
            size
            for name, (_algorithm, _identity, size) in self.file_pins.items()
            if name.startswith("model-") and name.endswith(".safetensors")
        )

    @property
    def shards(self) -> frozenset[str]:
        return frozenset(
            name for name in self.file_pins if name.startswith("model-") and name.endswith(".safetensors")
        )


FLASH_DECIDER = DeciderSourceProfile(
    family_id="decider-flash-v1.3",
    name="Decider Flash v1.3",
    repo="perplexity-ai/pplx-decider-flash-v1.3",
    revision="d06fb3f2d97e0a52325a0d463c450cf0338f44fe",
    base_model="Qwen/Qwen3.5-4B",
    tied_word_embeddings=True,
    parameter_count=4_539_265_536,
    shard_count=2,
    size_label="4B",
    file_pins={
        "LICENSE": ("git_blob_sha1", "f938136e3adacfd92be087f6e113b5d6d97f678f", 11544),
        "NOTICE": ("git_blob_sha1", "675e00d0e7554437fc0f139aeafa23c9a2b0ea56", 132),
        "chat_template.jinja": ("git_blob_sha1", "a585dec894e63da457d9440ec6aa7caa16d20860", 7756),
        "config.json": ("git_blob_sha1", "cc17ee553587bd774dd2d02415ea1ac0e0316104", 2898),
        "decision_config.json": ("git_blob_sha1", "c3302d2a1b3f5c7dc31cc866a3ac279be0295756", 4687),
        "joint_config.json": ("git_blob_sha1", "5f2b195316a2ed2dffcaa449746154f057c277b0", 273),
        "model-00001-of-00002.safetensors": (
            "sha256",
            "0048d1ddf2c5754fc05540fccdbc4ac50407e5cf14a35326be01cac0756b4de6",
            4972946624,
        ),
        "model-00002-of-00002.safetensors": (
            "sha256",
            "40e0c0ccb3248a59db0724ffd1ad1c81fc935d9f474e6fd52e9df32af4690608",
            4105669472,
        ),
        "model.safetensors.index.json": (
            "git_blob_sha1",
            "3d9b681bfe92f1e0d1cff660d6cecf9a199627fa",
            61898,
        ),
        "processor_config.json": ("git_blob_sha1", "424539fa61b2d9d23bf0e481ca3078e9ff404b12", 1219),
        "readout.safetensors": (
            "sha256",
            "46358a2e2639f1956561687ba261dd9b37506f878e90eb5eabc15f114532bec3",
            1305712,
        ),
        "tokenizer.json": (
            "sha256",
            "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            19989325,
        ),
        "tokenizer_config.json": (
            "git_blob_sha1",
            "875b7b0a3aabbd4e5e2888802dfbd5624a4a2060",
            1164,
        ),
    },
)

REGULAR_DECIDER = DeciderSourceProfile(
    family_id="decider-v1.3",
    name="Decider v1.3",
    repo="perplexity-ai/pplx-decider-v1.3",
    revision="9c2ebb27470d3dec2ad7ab727727e40d142c7213",
    base_model="Qwen/Qwen3.8-27B",
    tied_word_embeddings=False,
    parameter_count=26_085_330_160,
    shard_count=11,
    size_label="27B",
    file_pins={
        "LICENSE": ("sha256", "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a", 11544),
        "NOTICE": ("sha256", "d45742d08596d004a5d4a93630726d6961f70e653398d4ad2128979ecf1929f2", 127),
        "chat_template.jinja": (
            "sha256",
            "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
            8952,
        ),
        "config.json": ("sha256", "687f20bf493b8ee787dadaf338cf6df644dad44e32bcd2c3677a47c6c213d8b3", 3757),
        "decision_config.json": (
            "sha256",
            "4ee1c1e80044f6249c82e644b5334678670701b87f3972989c542ca86483e200",
            4688,
        ),
        "joint_config.json": (
            "sha256",
            "f7836e2e3fdaebaee8705e2ca25f5bb75bb146043ad530c84a6779a4af90801b",
            273,
        ),
        "model-00001-of-00011.safetensors": (
            "sha256",
            "683762f9ba120ed6980f6888d808c55812d1f3c54d79517aab7b0f5705ec7b20",
            4842451656,
        ),
        "model-00002-of-00011.safetensors": (
            "sha256",
            "b8a1694cc4ef7ae24e411adab0092e87a78abe547c97c47a8e19c57e345f1738",
            4965227416,
        ),
        "model-00003-of-00011.safetensors": (
            "sha256",
            "2ee7d8d814af369542d01b590f701c5f42e22ee756235a52125232f3d6e1a9b0",
            4912818744,
        ),
        "model-00004-of-00011.safetensors": (
            "sha256",
            "e853d09c8b2dedcb3f1a28d10f406df732d54d7603f55b1e6133e56deef8d3e5",
            4986198024,
        ),
        "model-00005-of-00011.safetensors": (
            "sha256",
            "87e712601578fccf91b0874c6972dd24ff1368e80d3075c0577eae956a07d637",
            4912818800,
        ),
        "model-00006-of-00011.safetensors": (
            "sha256",
            "a19864e0e8e77147ba8f0d0005dc70fca46f1521e24bcab2334a53492d2c4062",
            4932702808,
        ),
        "model-00007-of-00011.safetensors": (
            "sha256",
            "7a84ac88396740b1d02e88be50c4926bea99a3ec430dcac8b26738c8f1e4727a",
            4966314000,
        ),
        "model-00008-of-00011.safetensors": (
            "sha256",
            "413bd8c46e78ae179382511fdc0c0134a4db7d532f1981594b0fb6a55c0df36e",
            4964161752,
        ),
        "model-00009-of-00011.safetensors": (
            "sha256",
            "afb31b650ae67b9b271ee02a251e6ac5b775e811db0abe6084789d8920c83e6a",
            4933789312,
        ),
        "model-00010-of-00011.safetensors": (
            "sha256",
            "733da075428d6eb9ff0d6fc6e07caf482d70c1c86d77e4f37238fe4721be04f4",
            4965227504,
        ),
        "model-00011-of-00011.safetensors": (
            "sha256",
            "e68776c2fd0f3a503049fce03f4612850045a618819992af564967ed089878c3",
            2789092720,
        ),
        "model.safetensors.index.json": (
            "sha256",
            "0da01ffdc85b6f1bc596d3c38dd0d17663e18b6c81140269cbef28ea123f4b12",
            103954,
        ),
        "processor_config.json": (
            "sha256",
            "b4b905eee2d1cf96dcbdac1c68b35113c8151958a4c1f76667242b01168572cd",
            1219,
        ),
        "readout.safetensors": (
            "sha256",
            "46928c7cd0084289f6237a833720683c9d8ff91965706e746cc34b3ef0f12039",
            2611312,
        ),
        "tokenizer.json": (
            "sha256",
            "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            19989325,
        ),
        "tokenizer_config.json": (
            "sha256",
            "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
            1164,
        ),
    },
)

DECIDER_PROFILES = (FLASH_DECIDER, REGULAR_DECIDER)
DECIDER_BY_FAMILY = {profile.family_id: profile for profile in DECIDER_PROFILES}
DECIDER_BY_REPO = {profile.repo: profile for profile in DECIDER_PROFILES}
DECIDER_BY_REVISION = {profile.revision: profile for profile in DECIDER_PROFILES}


def decider_profile(
    value: str | None,
    *,
    revision: str | None = None,
) -> DeciderSourceProfile | None:
    if value in DECIDER_BY_FAMILY:
        profile = DECIDER_BY_FAMILY[value]
    elif value in DECIDER_BY_REPO:
        profile = DECIDER_BY_REPO[value]
    else:
        profile = DECIDER_BY_REVISION.get(value)
    if profile is not None and revision is not None and revision != profile.revision:
        return None
    return profile
