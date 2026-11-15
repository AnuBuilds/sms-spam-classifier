from pathlib import Path
import pickle
import runpy
from unittest.mock import patch

import pytest

from train import load_data, main


TRAIN_PATH = Path(__file__).resolve().parents[1] / "train.py"


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a small dataset for testing training behavior."""
    rows = []

    for number in range(10):
        rows.append(
            f"ham\tfamily dinner meeting library item{number}"
        )
        rows.append(
            f"spam\twinner prize cash claim bonus{number}"
        )

    # Add a duplicate to verify that loading removes it.
    rows.append(rows[0])

    data_path = tmp_path / "messages.tsv"
    data_path.write_text("\n".join(rows), encoding="utf-8")
    return data_path


def test_import_does_not_start_training():
    # The original script would try to read data during import.
    with patch("pandas.read_csv") as read_csv:
        runpy.run_path(str(TRAIN_PATH), run_name="import_check")

    read_csv.assert_not_called()


def test_loading_removes_duplicate_messages(sample_dataset):
    data = load_data(sample_dataset)

    assert len(data) == 20
    assert data["message"].is_unique
    assert data["label"].value_counts().to_dict() == {
        "ham": 10,
        "spam": 10,
    }


def test_training_saves_usable_model(sample_dataset, tmp_path):
    model_path = tmp_path / "test_model.pkl"

    model = main(
        data_path=sample_dataset,
        model_path=model_path,
    )

    assert model_path.is_file()
    assert model_path.stat().st_size > 0

    # This file was created by this test in a temporary directory.
    with model_path.open("rb") as file:
        saved_model = pickle.load(file)

    messages = [
        "family dinner meeting library",
        "winner prize cash claim",
    ]

    assert saved_model.predict(messages).tolist() == ["ham", "spam"]
    assert saved_model.predict(messages).tolist() == (
        model.predict(messages).tolist()
    )