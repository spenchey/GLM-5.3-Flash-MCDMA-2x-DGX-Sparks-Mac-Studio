import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep-mac-dflash-depth.py"
SPEC = importlib.util.spec_from_file_location("sweep_mac_dflash_depth", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_depth_list_is_bounded_and_unique():
    assert module._depths("1,3,7") == [1, 3, 7]
    for value in ("", "0", "8", "2,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            module._depths(value)


def test_configure_depth_changes_runtime_and_head_budget():
    class Head:
        nodes = 7

    class Runtime:
        drafts = 7
        head_drafts = Head()

    runtime = Runtime()
    module._configure_depth(runtime, 3)
    assert runtime.drafts == 3
    assert runtime.head_drafts.nodes == 3


def test_median_reads_numeric_run_fields():
    assert module._median([{"v": 3}, {"v": 1}, {"v": 2}], "v") == 2
