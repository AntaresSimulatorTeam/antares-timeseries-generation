import pytest
from pathlib import Path

from cluster_import import from_csv

@pytest.fixture
def data_directory() -> Path:
    return Path(__file__).parent / "data"

@pytest.fixture
def output_directory() -> Path:
    return Path(__file__).parent / "output"

@pytest.fixture
def cluster_1(data_directory) -> Path:
    return from_csv(data_directory / "cluster_1.csv")

@pytest.fixture
def cluster_100(data_directory) -> Path:
    return from_csv(data_directory / "cluster_100.csv")