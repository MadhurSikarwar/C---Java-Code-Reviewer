"""The engine's reasoning trace: real events, in order, through the API and the stream."""
import json, os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.bootstrap import models_present
pytestmark = pytest.mark.skipif(not models_present(), reason="trained models not built")

BAD = "#include <stdio.h>\nvoid f(void){\n  char b[8];\n  gets(b);\n}\n"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as c:
        yield c


def test_result_carries_an_ordered_trace(client):
    tr = client.post("/api/analyze", json={"code": BAD, "language": "C", "model_type": "v3"}).json()["trace"]
    stages = [e["stage"] for e in tr]
    assert stages[0] == "parse" and stages[-1] == "verdict"
    assert stages.index("findings") < stages.index("numbers") < stages.index("models") < stages.index("gate") < stages.index("verdict")
    assert [e["t"] for e in tr] == sorted(e["t"] for e in tr)
    finding = next(e for e in tr if e["kind"] == "finding")
    assert finding["line"] == 4 and finding["severity"] == "CRITICAL"
    model = next(e for e in tr if e["kind"] == "model")
    assert set(model["probs"]) == {"Clean", "Moderate", "High"} and sum(model["probs"].values()) in range(98, 103)
    assert any(e["kind"] == "counterfactual" for e in tr)          # findings exist, so "structure alone" is reported
    assert any(e["kind"] == "gate" and e["to"] == "High Risk" for e in tr)


def test_compare_all_traces_every_model(client):
    tr = client.post("/api/analyze", json={"code": BAD, "language": "C", "model_type": "all"}).json()["trace"]
    models = {e["model"] for e in tr if e["kind"] == "model"}
    from ml.predict import DL_MODEL_PATH
    if os.path.exists(DL_MODEL_PATH):
        assert models == {"ensemble", "dl", "v3", "v4_dl"}
    else:
        assert models == {"ensemble", "v3"}


def test_stream_is_ndjson_ending_with_the_result(client):
    r = client.post("/api/analyze/stream", json={"code": BAD, "language": "C", "model_type": "v3"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(l) for l in r.text.splitlines() if l.strip()]
    assert events[-1]["type"] == "result" and events[-1]["data"]["risk"]["label"] == "High Risk"
    assert all(e["type"] == "trace" for e in events[:-1]) and len(events) > 8


def test_stream_rejects_bad_input_with_a_normal_4xx(client):
    assert client.post("/api/analyze/stream", json={"code": "", "language": "C"}).status_code == 422
