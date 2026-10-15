import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


def audit_module():
    return importlib.import_module("scripts.audit")


@pytest.fixture
def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".audit/\n")
    (tmp_path / "source.txt").write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )
    return tmp_path


def test_audit_records_failure_and_continues_with_real_logs(repository):
    audit = audit_module()
    checks = [
        audit.Check("passes", [sys.executable, "-c", "print('worked')"]),
        audit.Check("fails", [sys.executable, "-c", "print('broken'); raise SystemExit(7)"]),
        audit.Check("later", [sys.executable, "-c", "print('still ran')"]),
    ]
    output = repository / ".audit" / "report.json"
    report = audit.run_audit(repository, checks, output, timeout_seconds=5)
    assert report == json.loads(output.read_text())
    assert report["schema_version"] == 1 and report["completed"] and not report["healthy"]
    assert [item["status"] for item in report["checks"]] == ["passed", "failed", "passed"]
    assert report["checks"][1]["returncode"] == 7
    assert Path(report["checks"][2]["log"]).read_text().strip() == "still ran"
    assert not report["source_changed"]
    assert report["source_start"]["dirty"] is False


def test_audit_publishes_running_and_not_run_before_executing(repository):
    audit = audit_module()
    output = repository / ".audit" / "report.json"
    code = (
        "import json,pathlib; data=json.loads(pathlib.Path('.audit/report.json').read_text()); "
        "assert not data['healthy'] and not data['completed']; "
        "assert [x['status'] for x in data['checks']] == ['running','not_run']"
    )
    report = audit.run_audit(
        repository,
        [
            audit.Check("inspect", [sys.executable, "-c", code]),
            audit.Check("after", [sys.executable, "-c", "pass"]),
        ],
        output,
        timeout_seconds=5,
    )
    assert report["healthy"]


def test_timeout_terminates_descendants_and_runs_remaining_checks(repository):
    audit = audit_module()
    code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',"
        "\"import pathlib,time; time.sleep(1); pathlib.Path('.audit/leaked').touch()\"]); "
        "time.sleep(10)"
    )
    report = audit.run_audit(
        repository,
        [
            audit.Check("timeout", [sys.executable, "-c", code]),
            audit.Check("after", [sys.executable, "-c", "pass"]),
        ],
        repository / ".audit/report.json",
        timeout_seconds=0.2,
    )
    assert [item["status"] for item in report["checks"]] == ["timed_out", "passed"]
    time.sleep(1.1)
    assert not (repository / ".audit/leaked").exists()
    assert not report["healthy"]


def test_source_change_invalidates_otherwise_passing_audit(repository):
    audit = audit_module()
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('source.txt').write_text('new')",
    ]
    report = audit.run_audit(
        repository,
        [audit.Check("mutates", command)],
        repository / ".audit/report.json",
        timeout_seconds=5,
    )
    assert report["checks"][0]["status"] == "passed"
    assert report["source_changed"] and not report["healthy"]
    assert report["source_end"]["fingerprint"] != report["source_start"]["fingerprint"]


def test_audit_refuses_reports_inside_unignored_source(repository):
    audit = audit_module()
    with pytest.raises(ValueError, match=r"ignored|outside"):
        audit.run_audit(
            repository,
            [audit.Check("one", [sys.executable, "-c", "pass"])],
            repository / "report.json",
            timeout_seconds=5,
        )
    assert not (repository / "report.json").exists()


@pytest.mark.parametrize("period", [0, -1, float("inf"), float("nan")])
def test_audit_rejects_unbounded_or_invalid_timeouts(repository, period):
    with pytest.raises(ValueError):
        audit_module().run_audit(
            repository, [], repository / ".audit/report.json", timeout_seconds=period
        )


def test_empty_audit_cannot_claim_success(repository):
    with pytest.raises(ValueError):
        audit_module().run_audit(
            repository, [], repository / ".audit/report.json", timeout_seconds=1
        )


def test_interrupted_audit_leaves_failure_and_unrun_evidence(repository):
    code = (
        "from pathlib import Path; import sys; from scripts.audit import Check, run_audit; "
        f"root=Path({str(repository)!r}); "
        'run_audit(root, [Check("interrupt", [sys.executable, "-c", '
        '"import os,signal,time; os.kill(os.getppid(),signal.SIGINT); time.sleep(10)"]), '
        'Check("later", [sys.executable,"-c","pass"])], '
        'root/".audit/report.json",timeout_seconds=5)'
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert process.returncode != 0
    report = json.loads((repository / ".audit/report.json").read_text())
    assert not report["completed"] and not report["healthy"]
    assert [check["status"] for check in report["checks"]] == ["interrupted", "not_run"]


def test_new_untracked_source_invalidates_audit(repository):
    audit = audit_module()
    code = "from pathlib import Path; Path('new.py').write_text('new source')"
    report = audit.run_audit(
        repository,
        [audit.Check("new-source", [sys.executable, "-c", code])],
        repository / ".audit/report.json",
        timeout_seconds=5,
    )
    assert report["source_changed"] and not report["healthy"]


def test_source_change_between_checks_remains_invalid_after_restore(repository):
    audit = audit_module()
    checks = [
        audit.Check(
            "change",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('source.txt').write_text('changed')",
            ],
        ),
        audit.Check(
            "restore",
            [
                sys.executable,
                "-c",
                "from pathlib import Path; assert Path('source.txt').read_text() == 'changed'; "
                "Path('source.txt').write_text('original')",
            ],
        ),
    ]
    report = audit.run_audit(
        repository, checks, repository / ".audit/report.json", timeout_seconds=5
    )
    assert all(check["status"] == "passed" for check in report["checks"])
    assert report["source_start"] == report["source_end"]
    assert report["source_changed"] and not report["healthy"]
    assert (
        report["checks"][0]["source_after"]["fingerprint"] != report["source_start"]["fingerprint"]
    )


def test_sigterm_interrupts_audit_and_kills_check_descendants(repository):
    command = (
        "import os,signal,subprocess,sys,time; "
        'subprocess.Popen([sys.executable,"-c",'
        "\"import pathlib,time; time.sleep(.4); pathlib.Path('.audit/leaked').touch()\"]); "
        "os.kill(os.getppid(),signal.SIGTERM); time.sleep(10)"
    )
    runner = (
        "from pathlib import Path; import signal,sys; from scripts.audit import Check,run_audit; "
        f"root=Path({str(repository)!r}); "
        f'checks=[Check("terminate",[sys.executable,"-c",{command!r}]),'
        'Check("later",[sys.executable,"-c","pass"])]; '
        "original=signal.getsignal(signal.SIGTERM); "
        '\ntry: run_audit(root,checks,root/".audit/report.json",timeout_seconds=.2)'
        "\nexcept KeyboardInterrupt: pass"
        "\nassert signal.getsignal(signal.SIGTERM) == original"
    )
    result = subprocess.run(
        [sys.executable, "-c", runner],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    time.sleep(0.6)
    assert not (repository / ".audit/leaked").exists()
    report = json.loads((repository / ".audit/report.json").read_text())
    assert not report["healthy"] and not report["completed"]
    assert [item["status"] for item in report["checks"]] == ["interrupted", "not_run"]
