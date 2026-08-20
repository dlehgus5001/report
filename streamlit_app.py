"""Streamlit dashboard for the before/after change-analysis demo."""

from __future__ import annotations

from collections import Counter
from html import escape
from io import BytesIO
from typing import Literal
from uuid import uuid4

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_drawable_canvas import st_canvas

from app.models import AnalysisResult, Box, Detection
from app.services import Pipeline

MAX_FILE_SIZE = 15 * 1024 * 1024
CHANGE_LABELS = {"new": "신규", "missing": "소실", "moved": "이동", "modified": "변경"}
DEFAULT_CLASSES = ["관심 객체", "사람", "차량", "건물", "시설물", "기타"]


def annotate_image(data: bytes, detections: list[Detection]) -> Image.Image:
    """Draw normalized detection boxes on an uploaded image."""
    image = Image.open(BytesIO(data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, min(width, height) // 180)
    font = ImageFont.load_default()

    for detection in detections:
        box = detection.box
        points = (
            int(box.x * width),
            int(box.y * height),
            int((box.x + box.width) * width),
            int((box.y + box.height) * height),
        )
        draw.rectangle(points, outline="#c9f36b", width=line_width)
        label = f"{detection.label} {detection.confidence:.0%}"
        label_box = draw.textbbox((points[0], points[1]), label, font=font)
        label_height = label_box[3] - label_box[1] + 8
        label_top = max(0, points[1] - label_height)
        draw.rectangle((points[0], label_top, label_box[2] + 8, points[1]), fill="#c9f36b")
        draw.text((points[0] + 4, label_top + 3), label, fill="#13221d", font=font)
    return image


def create_result(before: bytes, after: bytes, before_name: str, after_name: str) -> AnalysisResult:
    registration, detections, changes, report = Pipeline().run(before, after)
    counts = Counter(change.type.value for change in changes)
    return AnalysisResult(
        analysis_id=str(uuid4()),
        created_at="현재 세션",
        inputs={"before": before_name, "after": after_name},
        registration=registration,
        detections=detections,
        changes=changes,
        summary={kind: counts.get(kind, 0) for kind in CHANGE_LABELS},
        report=report,
    )


def validate_upload(file, label: str) -> bytes:
    data = file.getvalue()
    if len(data) > MAX_FILE_SIZE:
        raise ValueError(f"{label} 이미지는 15MB 이하여야 합니다.")
    try:
        Image.open(BytesIO(data)).verify()
    except Exception as exc:
        raise ValueError(f"{label} 파일을 이미지로 읽을 수 없습니다.") from exc
    return data


def add_manual_detection(
    result: AnalysisResult,
    image: Literal["before", "after"],
    label: str,
    confidence: float,
    x: float,
    y: float,
    width: float,
    height: float,
) -> Detection:
    """Append a user-reviewed box to an analysis result."""
    if x + width > 1 or y + height > 1:
        raise ValueError("박스가 이미지 경계를 벗어났습니다. 위치와 크기를 확인하세요.")
    detection = Detection(
        id=f"{image}-manual-{uuid4().hex[:8]}",
        image=image,
        label=label.strip() or "수동 객체",
        confidence=confidence,
        box=Box(x=x, y=y, width=width, height=height),
    )
    result.detections.append(detection)
    return detection


def delete_detection(result: AnalysisResult, detection_id: str) -> bool:
    """Delete one reviewed box by id and report whether it existed."""
    original_count = len(result.detections)
    result.detections = [item for item in result.detections if item.id != detection_id]
    return len(result.detections) != original_count


def detections_to_canvas(
    detections: list[Detection], canvas_width: int, canvas_height: int
) -> dict:
    """Convert normalized detections to Fabric.js rectangle objects."""
    return {
        "version": "4.4.0",
        "objects": [
            {
                "type": "rect",
                "left": item.box.x * canvas_width,
                "top": item.box.y * canvas_height,
                "width": item.box.width * canvas_width,
                "height": item.box.height * canvas_height,
                "fill": "rgba(201, 243, 107, 0.12)",
                "stroke": "#c9f36b",
                "strokeWidth": 3,
            }
            for item in detections
        ],
    }


def apply_canvas_boxes(
    result: AnalysisResult,
    image: Literal["before", "after"],
    objects: list[dict],
    canvas_width: int,
    canvas_height: int,
    new_label: str,
) -> list[Detection]:
    """Replace one image's detections with rectangles returned by the canvas."""
    previous = [item for item in result.detections if item.image == image]
    replacements: list[Detection] = []
    for index, item in enumerate(obj for obj in objects if obj.get("type") == "rect"):
        scale_x = float(item.get("scaleX", 1))
        scale_y = float(item.get("scaleY", 1))
        x = max(0.0, float(item.get("left", 0)) / canvas_width)
        y = max(0.0, float(item.get("top", 0)) / canvas_height)
        width = min(float(item.get("width", 0)) * scale_x / canvas_width, 1 - x)
        height = min(float(item.get("height", 0)) * scale_y / canvas_height, 1 - y)
        if width <= 0 or height <= 0:
            continue
        old = previous[index] if index < len(previous) else None
        replacements.append(
            Detection(
                id=old.id if old else f"{image}-manual-{uuid4().hex[:8]}",
                image=image,
                label=old.label if old else (new_label.strip() or "기타"),
                confidence=old.confidence if old else 1.0,
                box=Box(x=x, y=y, width=width, height=height),
            )
        )
    result.detections = [item for item in result.detections if item.image != image] + replacements
    return replacements


def render_box_editor(
    result: AnalysisResult, image: Literal["before", "after"], data: bytes
) -> None:
    """Render a mouse-driven canvas and a class-aware detection list."""
    title = "Before" if image == "before" else "After"
    detections = [item for item in result.detections if item.image == image]
    source = Image.open(BytesIO(data)).convert("RGB")
    canvas_width = 520
    canvas_height = max(180, round(canvas_width * source.height / source.width))
    background = source.resize((canvas_width, canvas_height))

    with st.expander(f"{title} 마우스 박스 편집 ({len(detections)}개)", expanded=True):
        mode = st.radio(
            "마우스 도구",
            ["박스 추가", "선택·이동·크기 조절"],
            horizontal=True,
            key=f"canvas_mode_{image}",
        )
        class_col, custom_col = st.columns(2)
        selected_class = class_col.selectbox(
            "새 박스 Class", DEFAULT_CLASSES, key=f"canvas_class_{image}"
        )
        custom_class = custom_col.text_input(
            "직접 입력 (선택)", placeholder="예: 굴착기", key=f"custom_class_{image}"
        )
        new_label = custom_class.strip() or selected_class
        st.caption(
            "이미지를 드래그해 박스를 추가하세요. 기존 박스는 선택 도구로 이동하거나 "
            "크기를 바꾼 뒤 ‘변경 적용’을 누르면 저장됩니다."
        )
        canvas = st_canvas(
            fill_color="rgba(201, 243, 107, 0.12)",
            stroke_width=3,
            stroke_color="#c9f36b",
            background_image=background,
            initial_drawing=detections_to_canvas(detections, canvas_width, canvas_height),
            drawing_mode="rect" if mode == "박스 추가" else "transform",
            display_toolbar=True,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            key=f"canvas_{image}_{st.session_state.get(f'canvas_version_{image}', 0)}",
        )
        if st.button("캔버스 변경 적용", key=f"apply_canvas_{image}"):
            objects = (canvas.json_data or {}).get("objects", [])
            apply_canvas_boxes(
                result, image, objects, canvas_width, canvas_height, new_label
            )
            st.session_state[f"canvas_version_{image}"] = (
                st.session_state.get(f"canvas_version_{image}", 0) + 1
            )
            st.rerun()

        st.markdown(f"#### 탐지 목록 · {len(detections)}개")
        if detections:
            for index, item in enumerate(detections, start=1):
                info, action = st.columns([4, 1])
                info.markdown(
                    f"**#{index} · Class: `{item.label}`**  \n"
                    f"신뢰도 {item.confidence:.0%} · "
                    f"x {item.box.x:.3f}, y {item.box.y:.3f}, "
                    f"w {item.box.width:.3f}, h {item.box.height:.3f}"
                )
                if action.button("삭제", key=f"delete_{image}_{item.id}"):
                    delete_detection(result, item.id)
                    st.session_state[f"canvas_version_{image}"] = (
                        st.session_state.get(f"canvas_version_{image}", 0) + 1
                    )
                    st.rerun()
        else:
            st.info("등록된 탐지 박스가 없습니다.")


st.set_page_config(page_title="Change Intelligence", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: #f2f0e8; color: #13221d;}
    [data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem;}
    .eyebrow {font-size:.72rem; letter-spacing:.18em; font-weight:800; color:#2e5b49;}
    .hero {font-family:Georgia,serif; font-size:clamp(2.6rem,6vw,5.2rem); line-height:.98;
           letter-spacing:-.04em; margin:.7rem 0 1rem;}
    .hero em {color:#2e5b49; font-style:normal;}
    .subcopy {color:#53605b; max-width:720px; line-height:1.7; margin-bottom:2rem;}
    div[data-testid="stFileUploader"] {background:#faf9f4; border:1px solid #cbc9bd;
      padding:1rem; border-radius:4px; min-height:155px;}
    div[data-testid="stMetric"] {background:#faf9f4; border:1px solid #cbc9bd; padding:1rem;}
    .report {background:#2e5b49; color:#f6f7f2; padding:1.5rem 1.7rem; border-radius:4px;
      line-height:1.8; min-height:170px; white-space:pre-line;}
    .demo-note {border-left:4px solid #ff7657; background:#fff7f3; padding:.8rem 1rem;
      color:#59443d; margin:1rem 0 1.5rem;}
    .stButton>button {background:#13221d; color:white; border:0; border-radius:2px;
      min-height:3.1rem; font-weight:700; width:100%;}
    .stButton>button:hover {background:#2e5b49; color:white; border:0;}
    </style>
    <div class="eyebrow">VISION ANALYSIS WORKSPACE · DEMO PIPELINE</div>
    <div class="hero">두 장면 사이의<br><em>의미 있는 변화</em>를 읽습니다.</div>
    <div class="subcopy">Before/After 이미지를 한 화면에서 비교하고, 정합·객체 탐지·변화 분류와
    AI 판독 초안을 순서대로 확인하세요.</div>
    """,
    unsafe_allow_html=True,
)

before_col, after_col = st.columns(2, gap="medium")
with before_col:
    st.subheader("01 · Before 기준 영상")
    before_file = st.file_uploader("Before 이미지", type=["png", "jpg", "jpeg", "webp"], key="before")
    if before_file:
        st.image(before_file, caption=before_file.name, use_container_width=True)
with after_col:
    st.subheader("02 · After 비교 영상")
    after_file = st.file_uploader("After 이미지", type=["png", "jpg", "jpeg", "webp"], key="after")
    if after_file:
        st.image(after_file, caption=after_file.name, use_container_width=True)

st.markdown('<div class="demo-note">현재는 화면과 전체 처리 흐름을 확인하는 데모입니다. 탐지 박스와 판독 내용은 실제 AI 모델 결과가 아닙니다.</div>', unsafe_allow_html=True)

if st.button("변화 분석 시작  →", type="primary", disabled=not (before_file and after_file)):
    try:
        before_data = validate_upload(before_file, "Before")
        after_data = validate_upload(after_file, "After")
        with st.spinner("이미지를 정합하고 변화 후보를 분석하고 있습니다…"):
            st.session_state.analysis = create_result(
                before_data, after_data, before_file.name, after_file.name
            )
            st.session_state.before_data = before_data
            st.session_state.after_data = after_data
    except ValueError as exc:
        st.error(str(exc))

if "analysis" in st.session_state:
    result = st.session_state.analysis
    st.divider()
    st.markdown('<div class="eyebrow">ANALYSIS RESULT</div>', unsafe_allow_html=True)
    st.header("변화 분석 결과")

    score_col, *change_cols = st.columns(5)
    score_col.metric("정합 신뢰도", f"{result.registration.score:.0%}")
    for column, (key, label) in zip(change_cols, CHANGE_LABELS.items()):
        column.metric(f"{label} 객체", result.summary[key])

    before_detections = [item for item in result.detections if item.image == "before"]
    after_detections = [item for item in result.detections if item.image == "after"]
    result_before, result_after = st.columns(2, gap="medium")
    with result_before:
        st.subheader("Before · 탐지 결과")
        st.image(annotate_image(st.session_state.before_data, before_detections), use_container_width=True)
    with result_after:
        st.subheader("After · 탐지 결과")
        st.image(annotate_image(st.session_state.after_data, after_detections), use_container_width=True)

    st.caption(
        "탐지 결과를 검수한 뒤 아래에서 박스를 추가하거나 삭제할 수 있습니다. "
        "변경 내용은 이미지와 JSON에 즉시 반영됩니다."
    )
    editor_before, editor_after = st.columns(2, gap="medium")
    with editor_before:
        render_box_editor(result, "before", st.session_state.before_data)
    with editor_after:
        render_box_editor(result, "after", st.session_state.after_data)

    changes_col, report_col = st.columns([1, 1.25], gap="medium")
    with changes_col:
        st.subheader("변화 목록")
        for change in result.changes:
            with st.container(border=True):
                st.markdown(f"**{CHANGE_LABELS[change.type.value]} · {change.label}**")
                st.write(change.description)
                st.caption(f"신뢰도 {change.confidence:.0%}")
    with report_col:
        st.subheader("AI 판독 보고서 · 초안")
        st.markdown(f'<div class="report">{escape(result.report)}</div>', unsafe_allow_html=True)

    with st.expander("구조화 데이터(JSON) 확인"):
        st.json(result.model_dump(mode="json"))
