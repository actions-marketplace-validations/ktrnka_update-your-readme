import pytest

from main import gha_escape, ReadmeRecommendation, fill_prompt


def test_gha_escape():
    assert gha_escape("test") == "test"
    assert gha_escape("test\n") == "test%0A"
    assert gha_escape("test\r") == "test%0D"
    assert gha_escape("test%") == "test%25"
    assert gha_escape("test\n\r%") == "test%0A%0D%25"


def test_output_validation():
    # test that it works normally
    ReadmeRecommendation(should_update=True, reason="test", updated_readme="test")

    # test that it fails with missing fields
    with pytest.raises(Exception):
        ReadmeRecommendation(should_update=True)  # type: ignore

    # test that it passes if should_update is False and the updated_readme is missing
    ReadmeRecommendation(should_update=False, reason="test", updated_readme=None)

    # test that it fails if should_update is True and the updated_readme is missing
    with pytest.raises(Exception):
        ReadmeRecommendation(should_update=True, reason="test")


def test_fill_prompt():
    """Test that fill_prompt doesn't crash"""
    assert "DEFAULT README" in str(fill_prompt("# DEFAULT README", "# PR STUFF"))
