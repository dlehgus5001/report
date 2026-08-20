from io import BytesIO

import pytest
from PIL import Image

from app.models import Box, Detection
from streamlit_app import add_manual_detection, annotate_image, create_result, delete_detection


def image_bytes(color: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (100, 80), color).save(output, format="PNG")
    return output.getvalue()


def test_annotation_keeps_image_size_and_draws_box():
    source = image_bytes("white")
    detection = Detection(
        id="before-object-1",
        image="before",
        label="object",
        confidence=0.9,
        box=Box(x=0.1, y=0.2, width=0.3, height=0.4),
    )
    annotated = annotate_image(source, [detection])
    assert annotated.size == (100, 80)
    assert annotated.getpixel((10, 16)) != (255, 255, 255)


def test_create_result_contains_both_image_detections():
    result = create_result(image_bytes("white"), image_bytes("gray"), "before.png", "after.png")
    assert {item.image for item in result.detections} == {"before", "after"}
    assert sum(result.summary.values()) == 1


def test_manual_detection_can_be_added_and_deleted():
    result = create_result(image_bytes("white"), image_bytes("gray"), "before.png", "after.png")
    detection = add_manual_detection(result, "before", "차량", 0.95, 0.1, 0.2, 0.3, 0.4)

    assert detection in result.detections
    assert detection.label == "차량"
    assert delete_detection(result, detection.id)
    assert detection not in result.detections
    assert not delete_detection(result, "unknown")


def test_manual_detection_rejects_box_outside_image():
    result = create_result(image_bytes("white"), image_bytes("gray"), "before.png", "after.png")

    with pytest.raises(ValueError, match="이미지 경계"):
        add_manual_detection(result, "after", "객체", 1.0, 0.8, 0.8, 0.3, 0.3)
