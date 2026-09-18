"""API tests (FastAPI TestClient). Needs the trained models: build them with `python ml/bootstrap.py --fast` if absent."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.bootstrap import models_present

pytestmark = pytest.mark.skipif(not models_present(), reason="trained models not built (run ml/bootstrap.py)")

C_BAD = "#include <stdio.h>\nvoid f(void){\n  char b[8];\n  gets(b);\n}\n"
C_OK = "#include <stdio.h>\nint main(void){ puts(\"hi\"); return 0; }\n"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as c:
        yield c


def post(client, code, lang="C", model="v3"):
    return client.post("/api/analyze", json={"code": code, "language": lang, "model_type": model})


def test_health_and_static_site(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "IntelliReview" in client.get("/").text
    assert client.get("/models.html").status_code == 200


def test_models_endpoint_serves_report(client):
    body = client.get("/api/models").json()
    assert {m["id"] for m in body["models"]} == {"v3", "dl", "ensemble", "v4_dl"}


def test_high_risk_and_line_and_fix(client):
    r = post(client, C_BAD).json()
    assert r["risk"]["label"] == "High Risk"
    issue = next(i for i in r["issues"] if i["type"] == "UNSAFE_FUNCTION")
    assert issue["line"] == 4 and "fgets(b, sizeof(b), stdin)" in issue["fix"]["after"]
    assert r["highlighted_lines"]["4"] == "CRITICAL"


def test_clean_code_is_clean(client):
    r = post(client, C_OK).json()
    assert r["risk"]["label"] == "Clean" and r["issues"] == []


@pytest.mark.parametrize("model", ["v3", "dl", "ensemble", "v4_dl", "all"])
def test_every_model_type_answers(client, model):
    r = post(client, C_BAD, model=model)
    assert r.status_code == 200 and r.json()["risk"]["label"] == "High Risk"


def test_compare_all_returns_four_opinions(client):
    assert set(post(client, C_BAD, model="all").json()["risk"]["comparisons"]) == {"ensemble", "dl", "v3", "v4_dl"}


def test_suppression_reported(client):
    r = post(client, C_BAD.replace("gets(b);", "gets(b); // intellireview: ignore")).json()
    assert r["issues"] == [] and len(r["suppressed"]) == 1


@pytest.mark.parametrize("payload,status", [
    ({"code": "", "language": "C"}, 422),
    ({"code": "hello world " * 30, "language": "C"}, 422),
    ({"code": "int main( { return ; }}}}", "language": "C"}, 422),
    ({"code": "int x;\x00", "language": "C"}, 422),
    ({"code": "int main(){}", "language": "Python"}, 422),
    ({"code": "int main(){}", "language": "C", "model_type": "../../etc/passwd"}, 422),
    ({"language": "C"}, 422),
    ({"code": "x" * 300_000, "language": "C"}, 413),
])
def test_bad_requests_are_4xx_never_5xx(client, payload, status):
    assert client.post("/api/analyze", json=payload).status_code == status


def test_java_sql_injection_end_to_end(client):
    code = ('import java.sql.*;\nclass A{ void q(Connection c) throws Exception{ java.io.BufferedReader r=new java.io.BufferedReader('
            'new java.io.InputStreamReader(System.in)); String u=r.readLine(); c.createStatement().executeQuery("SELECT * FROM t WHERE n=\'"+u+"\'"); } }')
    r = post(client, code, "Java").json()
    assert r["risk"]["label"] == "High Risk" and any(i["type"] == "SQL_INJECTION" for i in r["issues"])
