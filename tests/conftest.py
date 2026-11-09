import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _run_in_tmp_dir(tmp_path_factory):
    """build_recommendation_system writes runs/<run_id>/metrics.jsonl relative
    to the working directory; keep test runs out of the repo. Session-scoped
    so module-scoped fixtures that build a system are covered too. Data paths
    are absolute (DEFAULT_DATA_PATH), so nothing else depends on the cwd."""
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("cwd"))
    yield
    os.chdir(previous)
