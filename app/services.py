"""Replaceable boundaries for registration, detection, change analysis and LLM reporting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.models import Box, Change, ChangeType, Detection, Registration


class RegistrationModel:
    def run(self, before: bytes, after: bytes) -> Registration:
        # TODO: Replace with feature matching / learned registration output.
        similarity = 0.82 + (before[:16] == after[:16]) * 0.12
        return Registration(
            method="identity-demo",
            score=min(similarity, 0.99),
            transform=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )


class DetectionModel:
    def run(self, data: bytes, image: str) -> list[Detection]:
        # A stable placeholder makes the complete product flow testable before a model is installed.
        digest = hashlib.sha256(data).digest()
        offset = digest[0] / 2550
        return [
            Detection(
                id=f"{image}-object-1",
                image=image,
                label="관심 객체",
                confidence=round(0.78 + digest[1] / 2550, 3),
                box=Box(x=min(0.18 + offset, 0.35), y=0.22, width=0.28, height=0.24),
            )
        ]


class ChangeModel:
    def run(
        self, before: list[Detection], after: list[Detection], registration: Registration
    ) -> list[Change]:
        # TODO: Replace with matched-object association, pixel differences and change classifier.
        left, right = before[0], after[0]
        distance = abs(left.box.x - right.box.x)
        change_type = ChangeType.MOVED if distance > 0.025 else ChangeType.MODIFIED
        description = "객체 위치 변화가 감지되었습니다." if change_type == ChangeType.MOVED else "객체 속성 변화 후보가 감지되었습니다."
        return [
            Change(
                id="change-1",
                type=change_type,
                label=right.label,
                confidence=round(min(0.74 + registration.score / 10, 0.95), 3),
                description=description,
                before_box=left.box,
                after_box=right.box,
            )
        ]


class ReportModel:
    def run(self, changes: list[Change], registration: Registration) -> str:
        # TODO: Send the structured payload plus retrieved RAG context to an LLM.
        kinds = ", ".join(change.type.value for change in changes) or "없음"
        return (
            "자동 판독 초안\n\n"
            f"정합 신뢰도는 {registration.score:.0%}이며 변화 후보 {len(changes)}건이 확인되었습니다. "
            f"변화 유형은 {kinds}입니다. 본 결과는 데모 모델이 생성한 참고용 초안으로, "
            "실제 판독 전 원본 영상과 변화 영역을 전문가가 검토해야 합니다."
        )


@dataclass
class Pipeline:
    registration: RegistrationModel = RegistrationModel()
    detection: DetectionModel = DetectionModel()
    change: ChangeModel = ChangeModel()
    report: ReportModel = ReportModel()

    def run(self, before: bytes, after: bytes):
        registration = self.registration.run(before, after)
        detections = self.detection.run(before, "before") + self.detection.run(after, "after")
        changes = self.change.run(
            [item for item in detections if item.image == "before"],
            [item for item in detections if item.image == "after"],
            registration,
        )
        report = self.report.run(changes, registration)
        return registration, detections, changes, report

