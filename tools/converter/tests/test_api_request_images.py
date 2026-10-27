import base64

import pytest

from stark_convert.api_request import parse_chat_request
from stark_convert.local_api import APIError
from stark_convert.media import MAX_IMAGE_BYTES

PNG = b"\x89PNG\r\n\x1a\nsynthetic"


def image_part(data=PNG, *, detail=None, extra=None):
    value = {"url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")}
    if detail is not None:
        value["detail"] = detail
    if extra:
        value.update(extra)
    return {"type": "image_url", "image_url": value}


def parse(messages, *, allow_images=True):
    return parse_chat_request(
        {"messages": messages},
        max_output=32,
        vocab_size=256,
        allow_images=allow_images,
    )


def test_ordered_mixed_parts_keep_image_out_of_template_and_worker_messages():
    image = image_part()
    result = parse(
        [
            {"role": "developer", "content": "synthetic instruction"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    image,
                    {"type": "text", "text": "after"},
                ],
            },
        ]
    )
    assert result["messages"] == [
        {"role": "system", "content": "synthetic instruction"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "after"},
            ],
        },
    ]
    assert result["image"].data == PNG
    assert result["image"].media_type == "image/png"
    assert "data:image" not in repr(result["messages"])
    assert "synthetic" not in repr(result["image"])


def test_text_only_list_still_normalizes_to_string():
    result = parse([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert result["messages"] == [{"role": "user", "content": "ab"}]
    assert result["image"] is None


@pytest.mark.parametrize("role", ["assistant", "system"])
def test_image_parts_are_supported_only_on_user_messages(role):
    with pytest.raises(APIError) as error:
        parse([{ "role": role, "content": [image_part()] }])
    assert error.value.status == 400
    assert error.value.code == "images_unsupported"
    assert "data:image" not in str(error.value)


def test_default_text_mode_rejects_image_parts_before_dispatch():
    with pytest.raises(APIError) as error:
        parse([{"role": "user", "content": [image_part()]}], allow_images=False)
    assert error.value.status == 400
    assert error.value.code == "images_unsupported"


def test_only_one_image_across_all_messages_is_allowed():
    with pytest.raises(APIError) as error:
        parse(
            [
                {"role": "user", "content": [image_part()]},
                {"role": "user", "content": [image_part()]},
            ]
        )
    assert error.value.status == 400
    assert error.value.code == "too_many_images"


@pytest.mark.parametrize(
    ("part", "code"),
    [
        (image_part(detail="low"), "invalid_image_detail"),
        (image_part(detail="high"), "invalid_image_detail"),
        (image_part(extra={"filename": "private.png"}), "invalid_image_url"),
        (image_part(data=b"not a png"), "invalid_image"),
        ({"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}}, "invalid_image_url"),
    ],
)
def test_invalid_image_url_metadata_and_mime_are_fixed_client_errors(part, code):
    with pytest.raises(APIError) as error:
        parse([{"role": "user", "content": [part]}])
    assert error.value.status == 400
    assert error.value.code == code
    assert "private.png" not in str(error.value)
    assert "example.invalid" not in str(error.value)


def test_decoded_image_limit_is_enforced_at_api_parser():
    large_png = b"\x89PNG\r\n\x1a\n" + b"x" * (MAX_IMAGE_BYTES - 8 + 1)
    with pytest.raises(APIError) as error:
        parse([{"role": "user", "content": [image_part(large_png)]}])
    assert error.value.status == 413
    assert error.value.code == "image_too_large"


def test_text_budget_counts_utf8_bytes_across_messages():
    exact = "é" * (1024 * 1024 // 2)
    result = parse([{"role": "user", "content": exact}], allow_images=True)
    assert result["image"] is None
    assert len(result["messages"][0]["content"].encode("utf-8")) == 1024 * 1024
    with pytest.raises(APIError) as error:
        parse(
            [
                {"role": "user", "content": "é" * (1024 * 1024 // 4)},
                {"role": "assistant", "content": "é" * (1024 * 1024 // 4 + 1)},
            ],
            allow_images=True,
        )
    assert error.value.status == 413
    assert error.value.code == "text_too_large"
