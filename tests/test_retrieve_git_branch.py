import pytest
from cellacdc.myutils.app_info import get_git_branch_name

pytestmark = [pytest.mark.infra, pytest.mark.slow]

def test_get_git_branch_name():
    get_git_branch_name()