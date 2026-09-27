from typer.testing import CliRunner

from janus.cli import app


def test_doctor_runs():
    result = CliRunner().invoke(app, ["doctor"])
    assert "DOCTOR:" in result.output
