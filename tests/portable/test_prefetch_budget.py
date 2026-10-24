"""prefetch: a window's default share of video memory and RAM (Argus is shared)."""

import sys
import types

import pytest

GIB = 2**30


@pytest.fixture
def prefetch(monkeypatch):
    """cxngff.prefetch with ChimeraX's data cache stubbed (it is not importable here)."""
    for name in ("chimerax", "chimerax.map_data", "chimerax.map_data.datacache"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["chimerax.map_data.datacache"].Data_Cache = type("Data_Cache", (), {})
    monkeypatch.delitem(sys.modules, "cxngff.prefetch", raising=False)
    import cxngff.prefetch as mod

    monkeypatch.delenv("CHIMERAX_NGFF_GPU_GB", raising=False)
    monkeypatch.delenv("CHIMERAX_NGFF_PREFETCH_GB", raising=False)
    yield mod
    sys.modules.pop("cxngff.prefetch", None)


def test_gpu_cap_defaults_to_8_gb(prefetch):
    assert prefetch.gpu_cap_bytes() == 8 * GIB


def test_gpu_cap_env_overrides(prefetch, monkeypatch):
    monkeypatch.setenv("CHIMERAX_NGFF_GPU_GB", "16")
    assert prefetch.gpu_cap_bytes() == 16 * GIB
    monkeypatch.setenv("CHIMERAX_NGFF_GPU_GB", "0")
    assert prefetch.gpu_cap_bytes() == 0  # drawing ahead off
    monkeypatch.setenv("CHIMERAX_NGFF_GPU_GB", "lots")
    assert prefetch.gpu_cap_bytes() == 8 * GIB  # unparsable: the default


def test_cache_default_is_a_share_of_available_but_capped_at_16_gb(prefetch, monkeypatch):
    monkeypatch.setattr(prefetch, "available_bytes", lambda: 1000 * GIB)
    assert prefetch.budget_bytes() == 16 * GIB
    monkeypatch.setattr(prefetch, "available_bytes", lambda: 8 * GIB)
    assert prefetch.budget_bytes() == 6 * GIB  # 75 % of what is there
    monkeypatch.setattr(prefetch, "available_bytes", lambda: None)
    assert prefetch.budget_bytes() == 16 * GIB


def test_cache_env_overrides_the_cap(prefetch, monkeypatch):
    monkeypatch.setattr(prefetch, "available_bytes", lambda: 1000 * GIB)
    monkeypatch.setenv("CHIMERAX_NGFF_PREFETCH_GB", "64")
    assert prefetch.budget_bytes() == 64 * GIB
    monkeypatch.setenv("CHIMERAX_NGFF_PREFETCH_GB", "0")
    assert prefetch.budget_bytes() == 0  # prefetch off


def test_per_step_resize_keeps_the_cap(prefetch):
    """The Prefetcher re-reads the budget every step; it must not climb back to 75 %."""
    assert prefetch.default_cache_bytes(400 * GIB + 3 * GIB) == 16 * GIB
