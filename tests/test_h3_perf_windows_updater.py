"""Verify the Windows updater enables H3 diagnostics before starting SwarmUI.

CPU-only structural contract; Windows cmd --verify-h3-trace runs in CI separately.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
UPDATER = ROOT / "Windows_Update_SwarmUI_primeinc_test.bat"


def check_updater(text):
    norm = text.replace("\r\n", "\n")
    assert norm.startswith("@echo off\nsetlocal EnableExtensions\n")
    assert norm.count('set "SWARM_H3_TRACE=1"') == 1
    assert norm.index('set "SWARM_H3_TRACE=1"') < norm.index("call launch-windows.bat --port 7861")
    assert 'git switch dev || exit /b 1' in norm
    assert 'git switch -c dev --track origin/dev || exit /b 1' in norm
    assert 'git pull --ff-only || exit /b 1' in norm
    assert 'cmd /d /c set SWARM_H3_TRACE | findstr /X /C:"SWARM_H3_TRACE=1"' in norm
    assert 'if /I "%~1"=="--verify-h3-trace"' in norm
    assert 'pushd "%~dp0"' in norm and 'pushd "%~dp0.."' in norm
    assert norm.index("call launch-windows.bat --port 7861") < norm.index("popd\nexit /b %SWARM_LAUNCH_EXIT%")
    assert (ROOT / "tests" / "h3_perf_compare.py").is_file()


class H3WindowsUpdaterTests(unittest.TestCase):
    def test_dev_updater_enables_trace_before_launch_and_child_inheritance_probe(self):
        check_updater(UPDATER.read_text(encoding="utf-8"))

    def test_negative_control_trace_setting_removed_is_rejected(self):
        content = UPDATER.read_text(encoding="utf-8")
        mutant = content.replace('set "SWARM_H3_TRACE=1"', 'set "SWARM_H3_TRACE=0"', 1)
        with self.assertRaises(AssertionError):
            check_updater(mutant)

    def test_negative_control_wrong_branch_is_rejected(self):
        content = UPDATER.read_text(encoding="utf-8")
        mutant = content.replace("git switch dev || exit /b 1", "git switch master || exit /b 1", 1)
        with self.assertRaises(AssertionError):
            check_updater(mutant)


if __name__ == "__main__":
    unittest.main()
