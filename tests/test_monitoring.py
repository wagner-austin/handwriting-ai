from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pytest

import handwriting_ai.monitoring as mon
from handwriting_ai.logging import _JsonFormatter, get_logger

# Helper classes for mocking


class _DummyMemInfo:
    def __init__(self, rss: int) -> None:
        self.rss = rss


class _DummyVM:
    def __init__(self, total: int, available: int, used: int, percent: float) -> None:
        self.total = total
        self.available = available
        self.used = used
        self.percent = percent


class _DummyProcess:
    def __init__(self, pid: int, rss: int, children: list[_DummyProcess] | None = None) -> None:
        self.pid = pid
        self._rss = rss
        self._children = children or []

    def memory_info(self) -> _DummyMemInfo:
        return _DummyMemInfo(self._rss)

    def children(self, recursive: bool = False) -> list[_DummyProcess]:
        return self._children


# Test cgroup file I/O functions


def test_read_cgroup_file_success(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("12345\n", encoding="utf-8")
    result = mon._read_cgroup_file(test_file)
    assert result == "12345"


def test_read_cgroup_file_failure() -> None:
    nonexistent = Path("/nonexistent/file.txt")
    with pytest.raises(FileNotFoundError):
        mon._read_cgroup_file(nonexistent)


def test_read_cgroup_int_success(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("67890", encoding="utf-8")
    result = mon._read_cgroup_int(test_file)
    assert result == 67890


def test_read_cgroup_int_invalid(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("not_a_number", encoding="utf-8")
    with pytest.raises(ValueError):
        mon._read_cgroup_int(test_file)


def test_parse_cgroup_stat_valid() -> None:
    content = """anon 123456
file 789012
kernel 345678
slab 901234
some_other_field 111222
"""
    result = mon._parse_cgroup_stat(content)
    assert result["anon"] == 123456
    assert result["file"] == 789012
    assert result["kernel"] == 345678
    assert result["slab"] == 901234
    assert result["some_other_field"] == 111222


def test_parse_cgroup_stat_empty() -> None:
    content = ""
    result = mon._parse_cgroup_stat(content)
    assert result == {}


def test_parse_cgroup_stat_with_empty_lines() -> None:
    """Test continue on empty lines"""
    content = """anon 100

file 200

kernel 300
"""
    result = mon._parse_cgroup_stat(content)
    assert result == {"anon": 100, "file": 200, "kernel": 300}


def test_parse_cgroup_stat_malformed_format() -> None:
    """Test parsing skips lines with invalid format (not 2 parts)"""
    content = """anon 100
file 200 extra_field
kernel
slab 300
"""
    result = mon._parse_cgroup_stat(content)
    # Should only include valid lines (anon and slab)
    assert result == {"anon": 100, "slab": 300}


def test_parse_cgroup_stat_invalid_integer_raises_after_logging() -> None:
    """Test parsing raises on lines with non-integer values after logging"""
    content = """anon 100
file not_a_number
kernel 300
slab -invalid
total 400
"""
    import pytest

    # Should raise ValueError when encountering non-integer value
    with pytest.raises(ValueError):
        mon._parse_cgroup_stat(content)


# Test cgroup reading functions


def test_read_cgroup_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("524288000", encoding="utf-8")
    max_file.write_text("1048576000", encoding="utf-8")

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_MAX", max_file)

    usage = mon._read_cgroup_usage()
    assert usage.usage_bytes == 524288000
    assert usage.limit_bytes == 1048576000
    assert abs(usage.percent - 50.0) < 0.01


def test_read_cgroup_usage_unlimited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("524288000", encoding="utf-8")
    max_file.write_text("max", encoding="utf-8")

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_MAX", max_file)

    with pytest.raises(RuntimeError, match="unlimited"):
        mon._read_cgroup_usage()


def test_read_cgroup_usage_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current = tmp_path / "current"

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current)

    with pytest.raises(RuntimeError, match="no cgroup memory files found"):
        mon._read_cgroup_usage()


def test_read_cgroup_breakdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stat_file = tmp_path / "memory.stat"
    stat_content = """anon 100000000
file 200000000
kernel 50000000
slab 25000000
other_field 12345
"""
    stat_file.write_text(stat_content, encoding="utf-8")
    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat_file)

    breakdown = mon._read_cgroup_breakdown()
    assert breakdown.anon_bytes == 100000000
    assert breakdown.file_bytes == 200000000
    assert breakdown.kernel_bytes == 50000000
    assert breakdown.slab_bytes == 25000000


def test_read_cgroup_breakdown_missing_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stat_file = tmp_path / "memory.stat"
    stat_content = "other_field 12345\n"
    stat_file.write_text(stat_content, encoding="utf-8")
    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat_file)

    breakdown = mon._read_cgroup_breakdown()
    assert breakdown.anon_bytes == 0
    assert breakdown.file_bytes == 0
    assert breakdown.kernel_bytes == 0
    assert breakdown.slab_bytes == 0


def test_read_cgroup_breakdown_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stat = tmp_path / "stat"

    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat)

    with pytest.raises(RuntimeError, match=r"no cgroup memory\.stat file found"):
        mon._read_cgroup_breakdown()


def test_read_cgroup_breakdown_empty_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that empty stat file raises RuntimeError"""
    stat_file = tmp_path / "memory.stat"
    stat_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat_file)

    with pytest.raises(RuntimeError, match="parsing produced no valid entries"):
        mon._read_cgroup_breakdown()


def test_read_cgroup_breakdown_all_invalid_lines_raises_after_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that file with all invalid lines raises after logging"""
    stat_file = tmp_path / "memory.stat"
    stat_content = """invalid line format
field1 not_a_number
single_field
"""
    stat_file.write_text(stat_content, encoding="utf-8")
    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat_file)

    # Should raise ValueError from _parse_cgroup_stat after logging
    with pytest.raises(ValueError):
        mon._read_cgroup_breakdown()


def test_read_cgroup_breakdown_missing_core_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that missing core metrics (anon=0, file=0) logs warning but continues"""
    from io import StringIO

    stat_file = tmp_path / "memory.stat"
    stat_content = """kernel 50000000
slab 25000000
"""
    stat_file.write_text(stat_content, encoding="utf-8")
    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat_file)

    logger = get_logger()
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        breakdown = mon._read_cgroup_breakdown()
    finally:
        logger.removeHandler(handler)

    # Should return breakdown with zeros for anon and file
    assert breakdown.anon_bytes == 0
    assert breakdown.file_bytes == 0
    assert breakdown.kernel_bytes == 50000000
    assert breakdown.slab_bytes == 25000000

    # Should have logged warning
    out = buf.getvalue()
    assert "cgroup_breakdown_missing_core_metrics" in out


# Test worker process detection


def test_get_worker_processes_no_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[])

    def _proc_factory(pid: int) -> _DummyProcess:
        return parent

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)

    workers = mon._get_worker_processes(100)
    assert workers == ()


def test_get_worker_processes_with_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    child1 = _DummyProcess(pid=101, rss=50 * 1024 * 1024)
    child2 = _DummyProcess(pid=102, rss=75 * 1024 * 1024)
    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[child1, child2])

    def _proc_factory(pid: int) -> _DummyProcess:
        return parent

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)

    workers = mon._get_worker_processes(100)
    assert len(workers) == 2
    assert workers[0].pid == 101
    assert workers[0].rss_bytes == 50 * 1024 * 1024
    assert workers[1].pid == 102
    assert workers[1].rss_bytes == 75 * 1024 * 1024


def test_get_worker_processes_lookup_failure_raises_after_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that psutil.Process raises OSError after logging"""

    def _proc_factory(pid: int) -> None:
        raise OSError("Process not found")

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)
    # Should raise after logging
    with pytest.raises(OSError, match="Process not found"):
        mon._get_worker_processes(100)


def test_get_worker_processes_child_memory_failure_raises_after_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that child.memory_info() raises exception after logging"""

    class _FailingProcess:
        pid = 101

        def memory_info(self) -> None:
            raise OSError("Access denied")

        def children(self, recursive: bool = False) -> list[_FailingProcess]:
            return [self]

    def _proc_factory(pid: int) -> _FailingProcess:
        return _FailingProcess()

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)
    # Should raise after logging
    with pytest.raises(OSError, match="Access denied"):
        mon._get_worker_processes(100)


def test_get_worker_processes_invalid_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test line 185->179: isinstance check fails for non-int pid/rss"""

    class _BadMemInfo:
        rss = "not_an_int"  # Non-integer value

    class _BadProcess:
        pid = 101

        def memory_info(self) -> _BadMemInfo:
            return _BadMemInfo()

        def children(self, recursive: bool = False) -> list[_BadProcess]:
            return [self]

    def _proc_factory(pid: int) -> _BadProcess:
        return _BadProcess()

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)
    workers = mon._get_worker_processes(100)
    # Should skip the bad process and return empty tuple
    assert workers == ()


# Test CgroupMemoryMonitor


def test_cgroup_monitor_get_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Set up cgroup files
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    stat_file = tmp_path / "memory.stat"

    current_file.write_text("524288000", encoding="utf-8")
    max_file.write_text("1048576000", encoding="utf-8")
    stat_file.write_text(
        "anon 100000000\nfile 200000000\nkernel 50000000\nslab 25000000\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_MAX", max_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat_file)

    # Mock psutil for main process and workers
    child1 = _DummyProcess(pid=101, rss=50 * 1024 * 1024)
    child2 = _DummyProcess(pid=102, rss=75 * 1024 * 1024)
    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[child1, child2])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)
    monkeypatch.setattr("handwriting_ai.monitoring.os.getpid", lambda: 100)

    monitor = mon.CgroupMemoryMonitor()
    snap = monitor.get_snapshot()

    assert snap.main_process.pid == 100
    assert snap.main_process.rss_bytes == 100 * 1024 * 1024
    assert len(snap.workers) == 2
    assert snap.cgroup_usage.usage_bytes == 524288000
    assert snap.cgroup_usage.limit_bytes == 1048576000
    assert abs(snap.cgroup_usage.percent - 50.0) < 0.01
    assert snap.cgroup_breakdown.anon_bytes == 100000000
    assert snap.cgroup_breakdown.file_bytes == 200000000


def test_cgroup_monitor_check_pressure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("950000000", encoding="utf-8")
    max_file.write_text("1000000000", encoding="utf-8")

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_MAX", max_file)

    monitor = mon.CgroupMemoryMonitor()
    assert monitor.check_pressure(threshold_percent=90.0) is True
    assert monitor.check_pressure(threshold_percent=96.0) is False


def test_cgroup_monitor_log_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Set up minimal cgroup files
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    stat_file = tmp_path / "memory.stat"

    current_file.write_text("100000000", encoding="utf-8")
    max_file.write_text("200000000", encoding="utf-8")
    stat_file.write_text("anon 10000000\nfile 20000000\nkernel 5000000\nslab 2000000\n")

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_MAX", max_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_STAT", stat_file)

    parent = _DummyProcess(pid=100, rss=50 * 1024 * 1024, children=[])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)
    monkeypatch.setattr("handwriting_ai.monitoring.os.getpid", lambda: 100)

    monitor = mon.CgroupMemoryMonitor()
    logger = get_logger()
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        monitor.log_snapshot(context="test_context")
    finally:
        logger.removeHandler(handler)

    out = buf.getvalue()
    assert "test_context memory" in out
    assert "main_rss_mb=" in out
    assert "cgroup_usage_mb=" in out
    assert "cgroup_pct=" in out


# Test SystemMemoryMonitor


def test_system_monitor_get_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = _DummyProcess(pid=100, rss=100 * 1024 * 1024, children=[])

    def _proc_factory(pid: int | None = None) -> _DummyProcess:
        return parent

    def _vm() -> _DummyVM:
        return _DummyVM(
            total=16 * 1024 * 1024 * 1024,
            available=8 * 1024 * 1024 * 1024,
            used=8 * 1024 * 1024 * 1024,
            percent=50.0,
        )

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.Process", _proc_factory)
    monkeypatch.setattr("handwriting_ai.monitoring.psutil.virtual_memory", _vm)
    monkeypatch.setattr("handwriting_ai.monitoring.os.getpid", lambda: 100)

    monitor = mon.SystemMemoryMonitor()
    snap = monitor.get_snapshot()

    assert snap.main_process.pid == 100
    assert snap.main_process.rss_bytes == 100 * 1024 * 1024
    assert len(snap.workers) == 0
    assert snap.cgroup_usage.usage_bytes == 8 * 1024 * 1024 * 1024
    assert snap.cgroup_usage.limit_bytes == 16 * 1024 * 1024 * 1024
    assert abs(snap.cgroup_usage.percent - 50.0) < 0.01
    # Breakdown: anon approximated by main RSS, file from buffers+cached (0 if not present)
    assert snap.cgroup_breakdown.anon_bytes == 100 * 1024 * 1024
    assert snap.cgroup_breakdown.file_bytes == 0  # No buffers/cached in _DummyVM
    assert snap.cgroup_breakdown.kernel_bytes == 0
    assert snap.cgroup_breakdown.slab_bytes == 0


def test_system_monitor_check_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _vm() -> _DummyVM:
        return _DummyVM(
            total=100 * 1024 * 1024,
            available=5 * 1024 * 1024,
            used=95 * 1024 * 1024,
            percent=95.0,
        )

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.virtual_memory", _vm)

    monitor = mon.SystemMemoryMonitor()
    assert monitor.check_pressure(threshold_percent=90.0) is True
    assert monitor.check_pressure(threshold_percent=96.0) is False


# Test module-level functions and detection


def test_detect_cgroups_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current_file = tmp_path / "memory.current"
    current_file.write_text("1000", encoding="utf-8")

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)

    assert mon._detect_cgroups_available() is True


def test_detect_cgroups_not_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current = tmp_path / "current"

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current)

    assert mon._detect_cgroups_available() is False


def test_log_system_info_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Container mode: cgroup files exist
    current_file = tmp_path / "memory.current"
    max_file = tmp_path / "memory.max"
    current_file.write_text("500000000", encoding="utf-8")
    max_file.write_text("1000000000", encoding="utf-8")

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)
    monkeypatch.setattr(mon, "_CGROUP_MEM_MAX", max_file)

    def _cpu_count(logical: bool | None = True) -> int:
        return 8 if logical else 4

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.cpu_count", _cpu_count)

    logger = get_logger()
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        mon.log_system_info()
    finally:
        logger.removeHandler(handler)

    out = buf.getvalue()
    assert "system_info" in out
    assert "cpu_logical=8" in out
    assert "cpu_physical=4" in out
    assert "cgroup_mem_limit_mb=" in out


def test_log_system_info_non_container_raises_when_vm_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Non-container mode: no cgroup files
    current = tmp_path / "current"

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current)

    def _cpu_count(logical: bool | None = True) -> int:
        return 8 if logical else 4

    def _vm() -> _DummyVM:
        raise RuntimeError("VM info failed")

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.cpu_count", _cpu_count)
    monkeypatch.setattr("handwriting_ai.monitoring.psutil.virtual_memory", _vm)

    logger = get_logger()
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        # Should raise when virtual_memory fails
        with pytest.raises(RuntimeError, match="VM info failed"):
            mon.log_system_info()
    finally:
        logger.removeHandler(handler)


def test_create_monitor_cgroup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test _create_monitor returns CgroupMemoryMonitor"""
    current_file = tmp_path / "memory.current"
    current_file.write_text("1000", encoding="utf-8")

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current_file)

    monitor = mon._create_monitor()
    assert isinstance(monitor, mon.CgroupMemoryMonitor)


def test_create_monitor_system(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test _create_monitor returns SystemMemoryMonitor"""
    current = tmp_path / "current"

    monkeypatch.setattr(mon, "_CGROUP_MEM_CURRENT", current)

    def _vm() -> _DummyVM:
        return _DummyVM(
            total=16 * 1024 * 1024 * 1024,
            available=8 * 1024 * 1024 * 1024,
            used=8 * 1024 * 1024 * 1024,
            percent=50.0,
        )

    monkeypatch.setattr("handwriting_ai.monitoring.psutil.virtual_memory", _vm)

    monitor = mon._create_monitor()
    assert isinstance(monitor, mon.SystemMemoryMonitor)


def test_get_monitor() -> None:
    """Test line 336: get_monitor returns module-level singleton"""
    monitor = mon.get_monitor()
    assert monitor is mon._monitor
