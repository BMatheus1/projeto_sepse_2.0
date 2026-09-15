from __future__ import annotations

import argparse
import json
import math
import time
import hashlib
import importlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import (FINE_TUNING_DATASET_PATH, MOCK_FINETUNED_MODEL_PATH,
                     DEFAULT_BASE_MODEL, FINETUNED_DIR)

OPTIONAL_REAL_DEPENDENCIES = ["torch", "transformers", "datasets", "peft", "accelerate"]


def load_dataset(path: Path = FINE_TUNING_DATASET_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset de fine-tuning não encontrado: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if "messages" not in item or len(item["messages"]) < 3:
                    raise ValueError("Registro sem estrutura conversacional válida em messages.")
                rows.append(item)
    if not rows:
        raise ValueError("Dataset de fine-tuning está vazio.")
    return rows


def run_mock_finetuning(dataset_path: Path = FINE_TUNING_DATASET_PATH) -> Dict[str, Any]:
    rows = load_dataset(dataset_path)
    artifact = {
        "base_model": "modelo-base-simulado-llm-medico-academico",
        "examples": len(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "status": "mock_finetuning_completed",
        "note": "Simulação acadêmica reproduzível; nenhum modelo pesado foi treinado.",
    }
    MOCK_FINETUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MOCK_FINETUNED_MODEL_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = DEFAULT_BASE_MODEL
    epochs: float = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    max_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "learning_rate", "max_length", "lora_r", "lora_alpha"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deve ser positivo e finito.")
        if self.max_length < 32:
            raise ValueError("max_length deve ser pelo menos 32.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout deve estar entre 0 e 1.")


def check_real_dependencies() -> Dict[str, Any]:
    missing = []
    for package in OPTIONAL_REAL_DEPENDENCIES:
        try:
            importlib.import_module(package)
        except (ImportError, OSError):
            missing.append(package)
    if missing:
        return {
            "status": "real_finetuning_unavailable",
            "missing_dependencies": missing,
            "message": "Dependências ausentes ou incompatíveis. Instale requirements-finetuning.txt em Python 3.10–3.12 no Colab com CUDA. Nenhum treino foi executado.",
        }
    return {"status": "real_finetuning_ready", "message": "Dependências disponíveis; GPU ainda precisa ser verificada."}


def build_lora_config(config: TrainingConfig):
    from peft import LoraConfig
    return LoraConfig(task_type="CAUSAL_LM", r=config.lora_r,
                      lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
                      target_modules=["q_proj", "v_proj"], bias="none")


def build_training_metadata(config, rows, loss, elapsed, adapter_path, dataset_path):
    return {
        **asdict(config), "base_model": config.model_name, "dataset_size": len(rows),
        "train_loss": float(loss), "training_time_seconds": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "adapter_path": str(adapter_path.resolve()),
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "status": "real_finetuning_completed",
    }


def run_real_finetuning(config: TrainingConfig | None = None,
                        dataset_path: Path = FINE_TUNING_DATASET_PATH,
                        output_dir: Path = FINETUNED_DIR) -> Dict[str, Any]:
    config = config or TrainingConfig()
    config.validate()
    prerequisites = check_real_dependencies()
    if prerequisites["status"] != "real_finetuning_ready":
        raise RuntimeError(prerequisites["message"] + " Ausentes: " + ", ".join(prerequisites["missing_dependencies"]))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning real requer ambiente com GPU. Recomenda-se Google Colab com CUDA.")
    # Fail before any download; actual OOM is also handled below.
    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 2 * 1024**3:
        raise RuntimeError("Memória CUDA insuficiente: libere pelo menos 2 GiB; recomenda-se Colab T4. Modelos maiores exigem mais memória.")
    rows = load_dataset(dataset_path)
    output_dir = Path(output_dir)
    adapter_path = output_dir / "adapter"
    metadata_path = output_dir / "training_metadata.json"
    if adapter_path.exists() or metadata_path.exists():
        raise FileExistsError("Destino de treino já existe. Use --output-dir com outro diretório para preservar a evidência anterior.")
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                              TrainingArguments, DataCollatorForLanguageModeling, set_seed)
    from peft import get_peft_model
    started = time.perf_counter()
    try:
        set_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Qwen has a distinct pad token; EOS remains a training target.
        texts = [tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                               add_generation_prompt=False) for row in rows]
        dataset = Dataset.from_dict({"text": texts}).map(
            lambda batch: tokenizer(batch["text"], truncation=True,
                                    max_length=config.max_length, add_special_tokens=False),
            batched=True, remove_columns=["text"])
        bf16 = torch.cuda.is_bf16_supported()
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16 if bf16 else torch.float16)
        model.config.use_cache = False
        model = get_peft_model(model, build_lora_config(config))
        model.print_trainable_parameters()
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"), num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size, learning_rate=config.learning_rate,
            gradient_accumulation_steps=4, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bf16, fp16=not bf16, logging_steps=1, save_strategy="no",
            report_to="none", seed=42, dataloader_num_workers=0, optim="adamw_torch")
        trainer = Trainer(model=model, args=args, train_dataset=dataset,
                          processing_class=tokenizer,
                          data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
        result = trainer.train()
        # Trainer.training_loss is the mean over optimizer steps, not an evaluation metric.
        if not math.isfinite(result.training_loss) or result.global_step < 1:
            raise RuntimeError("Treinamento não produziu loss finita ou passos de otimização.")
        adapter_path.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(adapter_path, safe_serialization=True)
        tokenizer.save_pretrained(adapter_path)
        metadata = build_training_metadata(config, rows, result.training_loss,
                                           time.perf_counter() - started, adapter_path, dataset_path)
        metadata["last_logged_loss"] = next((x["loss"] for x in reversed(trainer.state.log_history) if "loss" in x), None)
        metadata["global_steps"] = result.global_step
        metadata["loss_history"] = trainer.state.log_history
        metadata["device"] = torch.cuda.get_device_name(0)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError("Memória CUDA insuficiente. Reduza --max-length/--batch-size ou use GPU Colab com mais memória. Treino não concluído.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning acadêmico da Fase 3.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Validação local; não treina pesos.")
    group.add_argument("--real", action="store_true", help="Executa treinamento LoRA em GPU CUDA.")
    for name, kind in [("model_name", str), ("epochs", float), ("batch_size", int),
                       ("learning_rate", float), ("max_length", int), ("lora_r", int), ("lora_alpha", int)]:
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(TrainingConfig(), name))
    parser.add_argument("--dataset-path", type=Path, default=FINE_TUNING_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=FINETUNED_DIR)
    args = parser.parse_args()
    try:
        config = TrainingConfig(**{name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__ if hasattr(args, name)})
        result = run_mock_finetuning(args.dataset_path) if args.mock else run_real_finetuning(config, args.dataset_path, args.output_dir)
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Fine-tuning não concluído: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
