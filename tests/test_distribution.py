"""Verify the files users actually receive, not just declared metadata."""

import subprocess
import tarfile
import zipfile
from pathlib import Path


def test_distributions_preserve_license_and_exclude_backups(tmp_path):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["uv", "build", "--no-sources", "--out-dir", str(tmp_path)],
        cwd=root,
        check=True,
    )
    expected = (root / "LICENSE").read_bytes()
    with zipfile.ZipFile(next(tmp_path.glob("*.whl"))) as wheel:
        names = wheel.namelist()
        licenses = [name for name in names if name.endswith(".dist-info/licenses/LICENSE")]
        assert len(licenses) == 1
        assert wheel.read(licenses[0]) == expected
        assert not any(name.endswith(("~", ".bak")) for name in names)
    with tarfile.open(next(tmp_path.glob("*.tar.gz"))) as source:
        names = source.getnames()
        licenses = [name for name in names if name.count("/") == 1 and name.endswith("/LICENSE")]
        assert len(licenses) == 1
        notice = source.extractfile(licenses[0])
        assert notice is not None and notice.read() == expected
        assert not any(name.endswith(("~", ".bak")) for name in names)
