import re
from pathlib import Path

import semantic_chunkers


def test_version_matches_pyproject():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M)
    assert match, "pyproject.toml has no version"
    assert semantic_chunkers.__version__ == match.group(1)
