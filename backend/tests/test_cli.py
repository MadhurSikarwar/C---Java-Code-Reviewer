"""Command line interface: report formats and exit codes."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cli

BAD = "#include <stdio.h>\nvoid f(void){\n  char b[8];\n  gets(b);\n}\n"
OK = '#include <stdio.h>\nint main(void){ puts("hi"); return 0; }\n'


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_exit_codes(tmp_path, capsys):
    bad, ok = write(tmp_path, "bad.c", BAD), write(tmp_path, "ok.c", OK)
    assert cli.main([ok]) == 0
    assert cli.main([bad]) == 1                      # default: fail on High
    assert cli.main([bad, "--fail-on", "none"]) == 0
    assert cli.main([str(tmp_path / "nothing_here")]) == 2


def test_fail_on_medium(tmp_path):
    p = write(tmp_path, "A.java", 'class A{ void f(){ try{ Integer.parseInt("x"); }catch(Exception e){} } }')
    assert cli.main([p]) == 0                        # a swallowed exception is Medium, below the default bar
    assert cli.main([p, "--fail-on", "medium"]) == 1


def test_sarif_is_valid_shape(tmp_path):
    bad = write(tmp_path, "bad.c", BAD)
    out = tmp_path / "r.sarif"
    cli.main([bad, "--format", "sarif", "-o", str(out), "--fail-on", "none"])
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["version"] == "2.1.0"
    res = d["runs"][0]["results"][0]
    assert res["ruleId"] == "UNSAFE_FUNCTION" and res["level"] == "error"
    assert res["locations"][0]["physicalLocation"]["region"]["startLine"] == 4


def test_directory_walk_skips_junk_and_suppression_applies(tmp_path):
    (tmp_path / "node_modules").mkdir()
    write(tmp_path / "node_modules", "x.c", BAD)
    write(tmp_path, "sup.c", BAD.replace("gets(b);", "gets(b); // intellireview: ignore"))
    assert cli.collect([str(tmp_path)]) == [str(tmp_path / "sup.c")]
    assert cli.main([str(tmp_path)]) == 0


def test_json_format(tmp_path, capsys):
    bad = write(tmp_path, "bad.c", BAD)
    cli.main([bad, "--format", "json", "--fail-on", "none"])
    data = json.loads(capsys.readouterr().out)
    assert data[0]["level"] == 2 and data[0]["issues"][0]["line"] == 4
