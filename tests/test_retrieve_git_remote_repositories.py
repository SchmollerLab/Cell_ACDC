import pytest
from cellacdc.myutils import get_git_remote_repos_names

pytestmark = pytest.mark.slow

def test_get_git_remote_repos_names():
    get_git_remote_repos_names()