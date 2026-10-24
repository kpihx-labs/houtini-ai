"""CLI surface tests that do not load a detection model."""

from ai_detect import __version__
from ai_detect.cli import main


def test_version_prints_installed_distribution_version(capsys):
    try:
        main(["--version"])
    except SystemExit as result:
        assert result.code == 0
    assert capsys.readouterr().out.strip() == f"houtini-ai {__version__}"
