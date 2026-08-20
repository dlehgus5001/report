from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
PNG = b"\x89PNG\r\n\x1a\n" + b"demo-image-content"


def test_health():
    assert client.get("/api/health").json() == {"status": "ok", "pipeline": "demo"}


def test_home_includes_before_after_result_viewer():
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="result-before"' in response.text
    assert 'id="result-after"' in response.text


def test_analysis_returns_structured_pipeline_result():
    response = client.post(
        "/api/analyze",
        files={"before": ("before.png", PNG, "image/png"), "after": ("after.png", PNG + b"2", "image/png")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["pipeline_mode"] == "demo"
    assert len(body["detections"]) == 2
    assert len(body["changes"]) == 1
    assert sum(body["summary"].values()) == 1


def test_rejects_unsupported_file_type():
    response = client.post(
        "/api/analyze",
        files={"before": ("a.txt", b"bad", "text/plain"), "after": ("after.png", PNG, "image/png")},
    )
    assert response.status_code == 415
