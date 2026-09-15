import io
import json
from zipfile import ZipFile

import pytest

from src.tc_fase3.import_adapter_zip import import_adapter_zip


def archive(extra=None):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as zipped:
        zipped.writestr("backup/adapter/adapter_config.json", json.dumps({
            "base_model_name_or_path": "test-only", "peft_type": "LORA"}))
        zipped.writestr("backup/adapter/adapter_model.safetensors", b"test-fixture-only")
        zipped.writestr("backup/adapter/tokenizer.json", "{}")
        for name, value in (extra or {}).items():
            zipped.writestr(name, value)
    return stream.getvalue()


def test_import_preserves_bytes_and_existing_destination(tmp_path):
    dest = tmp_path / "adapter"
    import_adapter_zip(archive(), dest)
    assert (dest / "adapter_model.safetensors").read_bytes() == b"test-fixture-only"
    assert (dest / "tokenizer.json").read_text() == "{}"
    with pytest.raises(FileExistsError):
        import_adapter_zip(archive(), dest)
    assert (dest / "adapter_model.safetensors").read_bytes() == b"test-fixture-only"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "backup/../../escape"])
def test_reject_unsafe_archive_without_writing(tmp_path, name):
    dest = tmp_path / "adapter"
    with pytest.raises(ValueError):
        import_adapter_zip(archive({name: "bad"}), dest)
    assert not dest.exists()


def test_reject_multiple_adapters(tmp_path):
    with pytest.raises(ValueError, match="exatamente um"):
        import_adapter_zip(archive({"other/adapter_config.json": "{}",
                                   "other/adapter_model.bin": "test"}), tmp_path / "adapter")


def test_reject_missing_weights(tmp_path):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as zipped:
        zipped.writestr("adapter_config.json", "{}")
    with pytest.raises(ValueError):
        import_adapter_zip(stream.getvalue(), tmp_path / "adapter")


@pytest.mark.parametrize("completed", [False, True])
def test_optional_notebook_comparison_preserves_report_without_loading_models(tmp_path, monkeypatch, capsys, completed):
    from pathlib import Path
    from src.tc_fase3 import config, evaluate_finetuned_model as evaluator, fine_tuned_llm
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebook/fase3_finetuning_colab.ipynb").read_text(encoding="utf-8"))
    cell = next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and "UPLOAD_ADAPTER_01 = False" in "".join(c["source"]))
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "REPORTS_FASE3_DIR", tmp_path)
    monkeypatch.setattr(fine_tuned_llm, "has_real_adapter", lambda path: False)
    def forbidden(**kwargs):
        pytest.fail("Should not run model evaluation")
    monkeypatch.setattr(evaluator, "compare_experiments", forbidden)
    report = tmp_path / "fine_tuning_comparison_experiments.json"
    content = json.dumps({"status": "real_models_evaluated" if completed else "not_evaluated_missing_adapters",
                          "missing_adapters": [] if completed else ["experiment_01"]})
    report.write_text(content)
    exec(compile(cell, "optional_comparison_cell", "exec"), {})
    assert report.read_text() == content
    output = capsys.readouterr().out
    assert ("Resultado existente preservado" if completed else "adapter do Experimento 01 ausente") in output
