import base64
import os
import stat

import pytest

from stark_convert.media import (
    MAX_IMAGE_BYTES,
    ImageData,
    MediaError,
    PrivateMediaStore,
    parse_image_url,
)

PNG = b"\x89PNG\r\n\x1a\nfixture"
JPEG = b"\xff\xd8\xfffixture\xff\xd9"


def data_url(data, subtype="png"):
    encoded = base64.b64encode(data).decode("ascii")
    return {"url": f"data:image/{subtype};base64,{encoded}"}


@pytest.mark.parametrize(
    ("payload", "media_type", "expected"),
    [(PNG, "png", "image/png"), (JPEG, "jpeg", "image/jpeg")],
)
def test_accepts_supported_inline_images(payload, media_type, expected):
    parsed = parse_image_url(data_url(payload, media_type))
    assert parsed.data == payload
    assert parsed.media_type == expected
    assert "data" not in repr(parsed)
    assert payload.decode("latin1") not in repr(parsed)


def test_accepts_only_omitted_or_auto_detail():
    item = data_url(PNG)
    item["detail"] = "auto"
    assert parse_image_url(item).data == PNG


@pytest.mark.parametrize("detail", ["low", "high", "AUTO", None, 1, {}, []])
def test_rejects_unsupported_detail(detail):
    item = data_url(PNG)
    item["detail"] = detail
    with pytest.raises(MediaError) as error:
        parse_image_url(item)
    assert error.value.code == "invalid_image_detail"
    assert "data:image" not in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        "https://example.invalid/image.png",
        {"url": "https://example.invalid/image.png"},
        {"url": "data:image/gif;base64,AAAA"},
        {"url": "data:image/png;charset=utf-8;base64,AAAA"},
        {"url": "data:image/png;base64,AA A"},
        {"url": "data:image/png;base64,é"},
        {"url": "data:image/png;base64,"},
        {"url": "data:image/png;base64,AAAA", "filename": "private.png"},
        {"url": "data:image/png;base64,AAAA", "hash": "private"},
        {"url": "data:image/png;base64,AAAA", "other": "metadata"},
        {"url": 1},
        [],
    ],
)
def test_rejects_untrusted_or_malformed_image_urls(value):
    with pytest.raises(MediaError) as error:
        parse_image_url(value)
    assert error.value.code in {"invalid_image_url", "invalid_image"}
    assert "private.png" not in str(error.value)


@pytest.mark.parametrize(
    "encoded",
    ["AQ==", "AQ===", "AB==", "AA==\n", "A===", "AAAA="],
)
def test_rejects_noncanonical_or_invalid_base64(encoded):
    with pytest.raises(MediaError) as error:
        parse_image_url({"url": f"data:image/png;base64,{encoded}"})
    assert error.value.code in {"invalid_image", "invalid_image_url"}


@pytest.mark.parametrize(
    "item",
    [
        {"url": f"data:image/png;base64,{base64.b64encode(JPEG).decode()}"},
        {"url": f"data:image/jpeg;base64,{base64.b64encode(PNG).decode()}"},
    ],
)
def test_rejects_mime_magic_mismatch(item):
    with pytest.raises(MediaError) as error:
        parse_image_url(item)
    assert error.value.code == "invalid_image"


def test_oversize_encoded_input_is_rejected_before_decode(monkeypatch):
    import stark_convert.media as media

    def forbidden_decode(*_args, **_kwargs):
        raise AssertionError("base64 decoder should not run for oversized input")

    monkeypatch.setattr(media.base64, "b64decode", forbidden_decode)
    oversized = "A" * (media._MAX_ENCODED_BYTES + 1)
    with pytest.raises(MediaError) as error:
        parse_image_url({"url": "data:image/png;base64," + oversized})
    assert error.value.code == "image_too_large"
    assert error.value.status == 413


def test_decoded_size_limit_is_enforced():
    payload = PNG + b"x" * (MAX_IMAGE_BYTES - len(PNG) + 1)
    encoded = base64.b64encode(payload).decode("ascii")
    with pytest.raises(MediaError) as error:
        parse_image_url({"url": "data:image/png;base64," + encoded})
    assert error.value.code == "image_too_large"


def test_private_store_stages_opaque_regular_file_then_unlinks(tmp_path):
    image = ImageData(PNG, "image/png")
    with PrivateMediaStore() as store:
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
        with store.stage(image) as metadata:
            assert set(metadata) == {"id", "byte_length", "media_type"}
            identifier = metadata["id"]
            assert len(identifier) == 32 and all(char in "0123456789abcdef" for char in identifier)
            path = store.root / identifier
            assert path.is_file() and not path.is_symlink()
            assert stat.S_ISREG(path.stat().st_mode)
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert path.read_bytes() == PNG
            assert metadata["byte_length"] == len(PNG)
            assert metadata["media_type"] == "image/png"
        assert not path.exists()
        assert list(store.root.iterdir()) == []
    assert not store.root.exists()


def test_private_store_unlinks_after_consumer_failure():
    with PrivateMediaStore() as store:
        with pytest.raises(RuntimeError):
            with store.stage(ImageData(PNG, "image/png")) as metadata:
                path = store.root / metadata["id"]
                raise RuntimeError("synthetic consumer failure")
        assert not path.exists()


def test_private_store_removes_partial_file_after_write_failure(monkeypatch):
    with PrivateMediaStore() as store:
        def fail_write(*_args):
            raise OSError("private path detail")

        monkeypatch.setattr(os, "write", fail_write)
        with pytest.raises(MediaError) as error:
            with store.stage(ImageData(PNG, "image/png")):
                raise AssertionError("stage should fail before yielding")
        assert error.value.code == "media_store_unavailable"
        assert "private path" not in str(error.value)
        assert list(store.root.iterdir()) == []


def test_private_store_does_not_follow_existing_symlink(monkeypatch, tmp_path):
    with PrivateMediaStore() as store:
        identifier = "a" * 32
        target = tmp_path / "outside"
        target.write_bytes(b"untouched")
        link = store.root / identifier
        link.symlink_to(target)
        monkeypatch.setattr("stark_convert.media.secrets.token_hex", lambda _n: identifier)
        with pytest.raises(MediaError) as error:
            with store.stage(ImageData(PNG, "image/png")):
                raise AssertionError("stage should not replace a symlink")
        assert error.value.code == "media_store_unavailable"
        assert target.read_bytes() == b"untouched"
        assert not link.exists()


def test_private_store_close_is_idempotent_and_rejects_new_stages():
    store = PrivateMediaStore()
    root = store.root
    store.close()
    store.close()
    assert not root.exists()
    with pytest.raises(MediaError) as error:
        with store.stage(ImageData(PNG, "image/png")):
            pass
    assert error.value.code == "media_store_unavailable"
