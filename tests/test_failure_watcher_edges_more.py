from __future__ import annotations

import handwriting_ai.jobs.watcher.logic as logic


def test_summarize_exc_info_detects_sigkill_and_waitpid() -> None:
    msg1 = "Worker killed; received SIGKILL (signal 9) during processing\nMore..."
    out1 = logic.summarize_exc_info(msg1)
    assert "OOM kill detected" in out1 or "SIGKILL" in out1

    msg2 = "Work-horse terminated unexpectedly; waitpid returned 9; details.."
    out2 = logic.summarize_exc_info(msg2)
    assert "Worker killed by OS" in out2


def test_coerce_job_ids_handles_bad_bytes_and_success() -> None:
    bad = b"\xff\xfe\xff"  # invalid UTF-8 should be skipped
    good = b"ok"
    out = logic.coerce_job_ids([bad, good, "already_str"])
    assert out == ["ok", "already_str"]


def test_make_default_ports() -> None:
    from handwriting_ai.jobs.watcher.ports import make_default_ports

    p = make_default_ports()
    # spot-check a couple of callables
    assert callable(p.redis_from_url)
    assert callable(p.rq_connect)
