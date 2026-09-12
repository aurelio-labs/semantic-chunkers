from importlib.metadata import version

import semantic_chunkers


def test_version_matches_installed_metadata():
    assert semantic_chunkers.__version__ == version("semantic-chunkers")
