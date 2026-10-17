import pytest


@pytest.fixture(scope="session")
def export_dir(project):
    return project.export
