"""Generated projects must run independently and remain editable."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from flybrain.__main__ import main
from flybrain.projects import create_project, doctor

ROOT = Path(__file__).resolve().parents[1]


def run_app(project, output):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(project / "app.py"), "--frames", "50", "--output", str(output)],
        cwd=project.parent,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )


@pytest.mark.parametrize("template", ["dodge", "tones"])
@pytest.mark.parametrize("model", ["toy", "real"])
def test_generated_project_runs_and_mapping_edits_change_output(tmp_path, template, model):
    reference = "toy" if model == "toy" else ROOT / "models/male-cns-escape-v1/model.json"
    project = create_project(tmp_path / "my demo", template, model=reference)
    before = tmp_path / "first"
    assert json.loads(run_app(project, before).stdout)["verified_replay"]
    recipe_path = project / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    for channel in recipe["readout"]["channels"].values():
        channel["weights"] = {neuron: 0 for neuron in channel["weights"]}
    recipe_path.write_text(json.dumps(recipe))
    after = tmp_path / "edited"
    run_app(project, after)
    frames = json.loads((before / "recording.json").read_text())["frames"]
    muted = json.loads((after / "recording.json").read_text())["frames"]
    assert any(any(f["requested"].values()) for f in frames)
    assert all(not any(f["requested"].values()) for f in muted)
    assert (after / "replay.html").is_file()
    assert (after / "session.json").is_file()
    if template == "tones":
        assert (after / "tones.wav").is_file()
    with pytest.raises(subprocess.CalledProcessError):
        run_app(project, after)  # Do not replace a previous recording.


def test_init_does_not_replace_existing_files(tmp_path):
    sentinel = tmp_path / "app.py"
    sentinel.write_text("keep me")
    with pytest.raises(FileExistsError):
        create_project(tmp_path)
    assert sentinel.read_text() == "keep me"


def test_project_checks_model_identity(tmp_path):
    project = create_project(tmp_path / "project")
    path = project / "recipe.json"
    recipe = json.loads(path.read_text())
    recipe["model_fingerprint"] = "wrong"
    path.write_text(json.dumps(recipe))
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_app(project, tmp_path / "run")
    assert "Model changed" in error.value.stderr
    assert not (tmp_path / "run").exists()


def test_cli_preserves_catalog_and_reports_capabilities(capsys, tmp_path):
    main(["models", "list"])
    assert "male-cns" in capsys.readouterr().out
    main(["demos", "list"])
    assert "tones" in capsys.readouterr().out
    main(["init", str(tmp_path / "project"), "--template", "tones"])
    assert "Created" in capsys.readouterr().out
    main(["demos", "run", "dodge", "--frames", "20", "--output", str(tmp_path / "run")])
    assert json.loads(capsys.readouterr().out)["verified_replay"]
    report = doctor()
    assert report["available_backends"]["cpu"] is True
    assert report["cpu_smoke"]["tick"] == 100
    assert report["cpu_smoke"]["action"]["walk"] > 0
    assert report["network_used"] is False


def test_cli_rejects_bad_frame_count_before_writing(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["demos", "run", "tones", "--frames", "0", "--output", str(tmp_path / "bad")])
    assert error.value.code == 2
    assert not (tmp_path / "bad").exists()
