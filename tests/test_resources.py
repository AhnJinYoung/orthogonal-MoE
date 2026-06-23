from orthomoe.resources import (
    MemoryGuard,
    apply_resource_limits,
    build_load_memory_kwargs,
    host_memory_summary,
    process_rss_gb,
)


def test_process_rss_is_reasonable():
    rss = process_rss_gb()
    # A live Python process uses some RAM but nowhere near a terabyte; this
    # guards against the ru_maxrss unit bug (KiB vs bytes vs GiB).
    assert 0.0 < rss < 1024.0


def test_host_memory_summary_keys():
    summary = host_memory_summary()
    assert "rss_gb" in summary
    assert summary["rss_gb"] >= 0.0


def test_apply_resource_limits_sets_threads():
    resolved = apply_resource_limits({"resources": {"num_threads": 2}})
    assert resolved["num_threads"] == 2


def test_build_load_memory_kwargs_budget():
    # No budget -> no special kwargs (preserves default behaviour).
    assert build_load_memory_kwargs({}, {}) == {}
    # With a CPU budget -> a cpu cap + a disk offload folder.
    kwargs = build_load_memory_kwargs({}, {"max_cpu_ram_gb": 32, "offload_folder": "/tmp/orthomoe_off"})
    assert "max_memory" in kwargs and "cpu" in kwargs["max_memory"]
    assert kwargs["offload_folder"] == "/tmp/orthomoe_off"


def test_memory_guard_no_budget_is_noop():
    # Without a budget the guard must never raise.
    MemoryGuard(None).check("noop")
