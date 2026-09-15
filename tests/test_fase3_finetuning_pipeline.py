import importlib
import json
import sys
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock
from contextlib import nullcontext
import pytest

from src.tc_fase3 import train_finetune as ft
from src.tc_fase3.config import FINE_TUNING_DATASET_PATH, PROTOCOLS_DIR
from src.tc_fase3.generate_synthetic_finetuning_data import generate_records, generate_dataset, DISTRIBUTION
from src.tc_fase3.evaluate_finetuned_model import evaluate, evaluation_prompts, score_response, update_report


def test_dataset_count_distribution_determinism_and_sources(tmp_path):
    records = generate_records()
    assert len(records) == 340
    assert Counter(r["category"] for r in records) == DISTRIBUTION
    assert len({r["input"] for r in records}) == 340
    path = tmp_path / "synthetic.jsonl"
    generate_dataset(path)
    first = path.read_bytes()
    generate_dataset(path)
    assert path.read_bytes() == first
    assert len(ft.load_dataset(FINE_TUNING_DATASET_PATH)) >= 120
    for record in records:
        source = record["output"].split("Fonte: ")[-1].rstrip(".")
        assert (PROTOCOLS_DIR / source).exists()


def test_mock_preserves_contract_without_real_training(tmp_path, monkeypatch):
    artifact = tmp_path / "mock.json"
    monkeypatch.setattr(ft, "MOCK_FINETUNED_MODEL_PATH", artifact)
    monkeypatch.setattr(ft, "run_real_finetuning", Mock(side_effect=AssertionError("no training")))
    result = ft.run_mock_finetuning()
    assert result["status"] == "mock_finetuning_completed"
    assert result["examples"] >= 120
    assert "train_loss" not in json.loads(artifact.read_text(encoding="utf-8"))


def test_missing_dependencies_message(monkeypatch, tmp_path):
    monkeypatch.setattr(ft.importlib, "import_module", Mock(side_effect=ImportError("absent")))
    with pytest.raises(RuntimeError, match="requirements-finetuning.txt"):
        ft.run_real_finetuning(output_dir=tmp_path / "training")


@pytest.fixture
def training_stubs(monkeypatch):
    # Exercise orchestration without torch, model downloads, GPU or optimization.
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
        mem_get_info=lambda: (8 * 1024**3, 16 * 1024**3), is_bf16_supported=lambda: False,
        get_device_name=lambda _: "test GPU", empty_cache=Mock(), OutOfMemoryError=type("OOM", (RuntimeError,), {})),
        bfloat16="bf16", float16="fp16")
    tokenizer = Mock(pad_token_id=0)
    def chat_template(messages, tokenize=True, add_generation_prompt=False):
        tokens = []
        for message in messages:
            tokens += [1] + [ord(c) for c in message["role"]] + [2]
            # Compact deterministic stub; exact boundary behavior is covered with a tokenizer test.
            tokens += [3, 4, 5]
        if add_generation_prompt:
            tokens += [1] + [ord(c) for c in "assistant"] + [2]
        return tokens
    tokenizer.apply_chat_template.side_effect = chat_template
    dataset = Mock()
    model = Mock()
    model.save_pretrained.side_effect = lambda path, **kw: ((path / "adapter_config.json").write_text('{}'), (path / "adapter_model.safetensors").write_bytes(b'test-only'))
    trainer = Mock()
    trainer.train.return_value = SimpleNamespace(training_loss=1.25, global_step=32)
    trainer.model = model
    trainer.evaluate.return_value = {"eval_loss": 1.05}
    trainer.state.best_metric = 1.05
    trainer.state.best_model_checkpoint = "checkpoint-12"
    trainer.state.log_history = [{"loss": 1.4}, {"eval_loss": 1.2}, {"loss": 1.1}, {"eval_loss": 1.05}]
    transformers = SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=Mock(from_pretrained=Mock(return_value=model)),
        Trainer=Mock(return_value=trainer), TrainingArguments=Mock(),
        DataCollatorForSeq2Seq=Mock(), set_seed=Mock())
    peft = SimpleNamespace(LoraConfig=Mock(), get_peft_model=Mock(return_value=model))
    for name, value in {"torch": torch, "transformers": transformers, "peft": peft,
                        "datasets": SimpleNamespace(Dataset=Mock(from_list=Mock(return_value=dataset)))}.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(ft, "check_real_dependencies", lambda: {"status": "real_finetuning_ready"})
    return SimpleNamespace(torch=torch, transformers=transformers, peft=peft, trainer=trainer, tokenizer=tokenizer)


def test_real_orchestration_and_metadata_with_stubs(training_stubs, tmp_path):
    result = ft.run_real_finetuning(output_dir=tmp_path)
    training_stubs.trainer.train.assert_called_once()
    assert result["status"] == "real_finetuning_completed"
    for key in ["base_model", "dataset_size", "epochs", "learning_rate", "batch_size", "lora_r", "lora_alpha",
                "lora_dropout", "train_loss", "training_time_seconds", "timestamp", "adapter_path", "dataset_sha256"]:
        assert key in result
    assert result["train_loss"] == 1.25
    assert result["last_logged_loss"] == 1.1
    assert result["eval_loss"] == result["best_eval_loss"] == 1.05
    assert result["loss_masking"] == "assistant_only"
    assert result["dataset_size"] == 279 and result["validation_size"] == 35
    args = training_stubs.transformers.TrainingArguments.call_args.kwargs
    assert args["eval_strategy"] == args["save_strategy"] == "epoch"
    assert args["load_best_model_at_end"] is True
    assert "eval_dataset" in training_stubs.transformers.Trainer.call_args.kwargs
    assert (tmp_path / "training_metadata.json").exists()
    kwargs = training_stubs.peft.LoraConfig.call_args.kwargs
    assert kwargs["r"] == 16 and kwargs["lora_alpha"] == 32
    assert kwargs["task_type"] == "CAUSAL_LM"
    assert training_stubs.tokenizer.apply_chat_template.call_count >= 120


def test_cpu_stops_before_model_download(training_stubs, tmp_path):
    training_stubs.torch.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match="GPU"):
        ft.run_real_finetuning(output_dir=tmp_path)
    training_stubs.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
    assert not (tmp_path / "training_metadata.json").exists()


def test_oom_does_not_claim_success(training_stubs, tmp_path):
    training_stubs.trainer.train.side_effect = training_stubs.torch.cuda.OutOfMemoryError("full")
    with pytest.raises(RuntimeError, match="Memória CUDA insuficiente"):
        ft.run_real_finetuning(output_dir=tmp_path)
    assert not (tmp_path / "training_metadata.json").exists()
    assert not (tmp_path / "adapter").exists()


def test_retraining_does_not_overwrite_evidence(training_stubs, tmp_path):
    (tmp_path / "adapter").mkdir()
    with pytest.raises(FileExistsError, match="preservar"):
        ft.run_real_finetuning(output_dir=tmp_path)
    training_stubs.trainer.train.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"learning_rate": float("nan")}, {"max_length": 1}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ft.TrainingConfig(**kwargs).validate()


def test_evaluation_no_adapter_is_honest(tmp_path):
    factory = Mock(side_effect=AssertionError("must not load"))
    result = evaluate(tmp_path / "missing", output_dir=tmp_path, model_factory=factory)
    assert result["status"] == "not_evaluated_no_real_adapter"
    assert result["results"] == []
    assert "not_evaluated_no_real_adapter" in (tmp_path / "fine_tuning_evaluation.csv").read_text()
    assert len(evaluation_prompts()) >= 10


def test_base_vs_adapter_evaluation_and_report(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "test-base"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"test-only")
    model = Mock()
    model.generate.return_value = "Não posso confirmar doença. Exige validação humana. Fonte: protocolo_avaliacao_sintetico."
    factory = Mock(return_value=model)
    result = evaluate(adapter, output_dir=tmp_path, model_factory=factory)
    assert result["status"] == "real_models_evaluated"
    assert len(result["results"]) == 40
    assert factory.call_args_list[0].args == ("test-base", None)
    assert factory.call_args_list[1].args == ("test-base", adapter)
    assert not any(score_response("", evaluation_prompts()[0]).values())
    report = tmp_path / "report.md"
    report.write_text("# Preservar texto existente\n")
    update_report(result, tmp_path / "missing_metadata.json", report)
    update_report(result, tmp_path / "missing_metadata.json", report)
    text = report.read_text(encoding="utf-8")
    assert "Preservar" in text and "Pendente de execução" in text
    assert text.count("<!-- EXPERIMENT_02_RESULTS_START -->") == 1
