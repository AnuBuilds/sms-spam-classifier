from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_valid_message_returns_not_spam():
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception

    app.text_area[0].input(
        "Hey, are we still meeting at the library tomorrow at 10?"
    ).run()
    app.button[0].click().run()

    assert not app.exception
    assert len(app.success) == 1
    assert app.success[0].value == "Prediction: Not spam"
    assert len(app.warning) == 0


@pytest.mark.parametrize("message", ["", "   "])
def test_blank_message_shows_warning(message):
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception

    app.text_area[0].input(message).run()
    app.button[0].click().run()

    assert not app.exception
    assert len(app.warning) == 1
    assert app.warning[0].value == "Enter a message first."
    assert len(app.success) == 0
    assert len(app.error) == 0