import pytest


@pytest.fixture(autouse=True)
def feedback_db(tmp_path, monkeypatch):
    """Answers the API logs for feedback go to a temporary database, never data/feedback/."""
    path = tmp_path / "feedback.db"
    monkeypatch.setenv("FEEDBACK_DB", str(path))
    return path
