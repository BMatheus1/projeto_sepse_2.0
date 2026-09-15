from __future__ import annotations

import argparse
import json
import math
import time
import hashlib
import importlib
import inspect
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import (FINE_TUNING_DATASET_PATH, MOCK_FINETUNED_MODEL_PATH,
                     DEFAULT_BASE_MODEL, EXPERIMENT_02_DIR,
                     FINE_TUNING_TRAIN_PATH, FINE_TUNING_VALIDATION_PATH)

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
    epochs: float = 3
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_length: int = 512
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    seed: int = 42

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "learning_rate", "max_length", "lora_r", "lora_alpha", "gradient_accumulation_steps"):
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


def build_assistant_only_labels(input_ids, assistant_spans, attention_mask=None):
    """Mask system/user, assistant headers and padding; retain answer and EOS.

    Spans are half-open token offsets obtained from verified chat-template prefixes.
    No string search for 'assistant' that an adversarial user could inject is used.
    """
    if attention_mask is not None and len(attention_mask) != len(input_ids):
        raise ValueError("attention_mask deve ter o mesmo tamanho de input_ids.")
    labels = [-100] * len(input_ids)
    for start, end in assistant_spans:
        if not 0 <= start < end <= len(input_ids):
            raise ValueError("Intervalo assistant inválido.")
        for index in range(start, end):
            if attention_mask is None or attention_mask[index]:
                labels[index] = input_ids[index]
    if not any(label != -100 for label in labels):
        raise ValueError("Exemplo sem tokens assistant supervisionados.")
    return labels


def tokenize_conversation(row, tokenizer, max_length=512):
    messages = row["messages"]
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if not message["content"].strip():
            raise ValueError("Resposta assistant vazia.")
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        completed = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, add_generation_prompt=False)
        if full[:len(prefix)] != prefix or full[:len(completed)] != completed:
            raise ValueError("Chat template sem prefixo estável; masking recusado para não treinar tokens user/system.")
        spans.append((len(prefix), len(completed)))
    if len(full) > max_length:
        raise ValueError(f"Diálogo tem {len(full)} tokens e excede --max-length {max_length}. "
                         "Reduza o exemplo ou aumente o limite; a fonte e os limites não serão truncados silenciosamente.")
    mask = [1] * len(full)
    return {"input_ids": full, "attention_mask": mask,
            "labels": build_assistant_only_labels(full, spans, mask)}


def validate_train_validation_split(train_rows, validation_rows):
    from .prepare_finetuning_dataset import normalized_text
    def texts(rows, role):
        return {normalized_text(m["content"]) for row in rows for m in row["messages"] if m["role"] == role}
    for role in ("user", "assistant"):
        if texts(train_rows, role) & texts(validation_rows, role):
            raise ValueError("Vazamento entre treino e validação: texto duplicado.")
    groups = lambda rows: {r.get("metadata", {}).get("group_id") for r in rows} - {None}
    if groups(train_rows) & groups(validation_rows):
        raise ValueError("Vazamento entre treino e validação: famílias compartilhadas.")


def epoch_evaluation_kwargs(arguments_class):
    parameters = inspect.signature(arguments_class).parameters
    name = "evaluation_strategy" if "evaluation_strategy" in parameters and "eval_strategy" not in parameters else "eval_strategy"
    return {name: "epoch"}


def run_real_finetuning(config: TrainingConfig | None = None,
                        dataset_path: Path = FINE_TUNING_TRAIN_PATH,
                        output_dir: Path = EXPERIMENT_02_DIR,
                        validation_path: Path = FINE_TUNING_VALIDATION_PATH) -> Dict[str, Any]:
    config = config or TrainingConfig()
    config.validate()
    output_dir, dataset_path, validation_path = Path(output_dir), Path(dataset_path), Path(validation_path)
    adapter_path = output_dir / "adapter"
    metadata_path = output_dir / "training_metadata.json"
    if any((output_dir / name).exists() for name in ("adapter", "training_metadata.json", "checkpoints", "datasets")):
        raise FileExistsError("Destino de treino já existe. Use --output-dir com outro diretório para preservar a evidência anterior.")
    prerequisites = check_real_dependencies()
    if prerequisites["status"] != "real_finetuning_ready":
        raise RuntimeError(prerequisites["message"] + " Ausentes: " + ", ".join(prerequisites["missing_dependencies"]))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Fine-tuning real requer ambiente com GPU. Recomenda-se Google Colab com CUDA.")
    free_memory, _ = torch.cuda.mem_get_info()
    if free_memory < 2 * 1024**3:
        raise RuntimeError("Memória CUDA insuficiente: libere pelo menos 2 GiB; recomenda-se Colab T4.")
    rows, validation_rows = load_dataset(dataset_path), load_dataset(validation_path)
    validate_train_validation_split(rows, validation_rows)
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                              TrainingArguments, DataCollatorForSeq2Seq, set_seed)
    from peft import get_peft_model
    started = time.perf_counter()
    try:
        set_seed(config.seed)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        train_tokens = [tokenize_conversation(row, tokenizer, config.max_length) for row in rows]
        val_tokens = [tokenize_conversation(row, tokenizer, config.max_length) for row in validation_rows]
        dataset = Dataset.from_list(train_tokens)
        validation_dataset = Dataset.from_list(val_tokens)
        # Snapshot inputs before training. Test is archived only, never passed to Trainer.
        snapshot_dir = output_dir / "datasets"
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        data_info = {}
        for name, path in [("train", dataset_path), ("validation", validation_path),
                           ("test", dataset_path.parent / "fine_tuning_test.jsonl")]:
            if path.exists():
                snapshot = snapshot_dir / f"fine_tuning_{name}.jsonl"
                shutil.copyfile(path, snapshot)
                data_info[name] = {"path": str(snapshot.resolve()), "size": len(load_dataset(snapshot)),
                                   "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}
        bf16 = torch.cuda.is_bf16_supported()
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.bfloat16 if bf16 else torch.float16)
        model.config.use_cache = False
        model = get_peft_model(model, build_lora_config(config))
        model.print_trainable_parameters()
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"), num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size, per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate, gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=bf16, fp16=not bf16, logging_steps=1, save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            label_names=["labels"], prediction_loss_only=True, report_to="none", seed=config.seed,
            dataloader_num_workers=0, optim="adamw_torch", **epoch_evaluation_kwargs(TrainingArguments))
        trainer = Trainer(model=model, args=args, train_dataset=dataset, eval_dataset=validation_dataset,
                          processing_class=tokenizer,
                          data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100,
                                                              pad_to_multiple_of=8))
        result = trainer.train()
        eval_loss = float(trainer.evaluate()["eval_loss"])
        best_eval_loss = float(trainer.state.best_metric) if trainer.state.best_metric is not None else eval_loss
        if not all(math.isfinite(value) for value in (result.training_loss, eval_loss, best_eval_loss)) or result.global_step < 1:
            raise RuntimeError("Treinamento não produziu losses finitas ou passos de otimização.")
        adapter_path.mkdir(parents=True, exist_ok=False)
        # Trainer has restored the checkpoint with minimum validation loss.
        trainer.model.save_pretrained(adapter_path, safe_serialization=True)
        tokenizer.save_pretrained(adapter_path)
        metadata = build_training_metadata(config, rows, result.training_loss,
                                           time.perf_counter() - started, adapter_path, snapshot_dir / "fine_tuning_train.jsonl")
        metadata.update({"experiment": output_dir.name, "loss_masking": "assistant_only",
                         "eval_loss": eval_loss, "best_eval_loss": best_eval_loss,
                         "best_checkpoint": trainer.state.best_model_checkpoint,
                         "validation_size": len(validation_rows), "datasets": data_info,
                         "dataset_total_size": sum(info["size"] for info in data_info.values()),
                         "max_observed_tokens": max(len(row["input_ids"]) for row in train_tokens + val_tokens),
                         "supervised_train_tokens": sum(sum(label != -100 for label in row["labels"]) for row in train_tokens),
                         "last_logged_loss": next((x["loss"] for x in reversed(trainer.state.log_history) if "loss" in x), None),
                         "global_steps": result.global_step, "loss_history": trainer.state.log_history,
                         "device": torch.cuda.get_device_name(0)})
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError("Memória CUDA insuficiente. Reduza --max-length/--batch-size ou use GPU Colab com mais memória. Treino não concluído.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning acadêmico da Fase 3 — Experimento 02.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mock", action="store_true", help="Validação local; não treina pesos.")
    group.add_argument("--real", action="store_true", help="Executa treinamento LoRA em GPU CUDA.")
    for name, kind in [("model_name", str), ("epochs", float), ("batch_size", int),
                       ("learning_rate", float), ("max_length", int), ("lora_r", int), ("lora_alpha", int),
                       ("lora_dropout", float), ("gradient_accumulation_steps", int), ("seed", int)]:
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(TrainingConfig(), name))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--validation-path", type=Path, default=FINE_TUNING_VALIDATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_02_DIR)
    args = parser.parse_args()
    try:
        config = TrainingConfig(**{name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__})
        if args.mock:
            result = run_mock_finetuning(args.dataset_path or FINE_TUNING_DATASET_PATH)
        else:
            result = run_real_finetuning(config, args.dataset_path or FINE_TUNING_TRAIN_PATH,
                                         args.output_dir, args.validation_path)
    except (RuntimeError, OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"Fine-tuning não concluído: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
