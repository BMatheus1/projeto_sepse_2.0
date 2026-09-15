"""Backend local HF/PEFT, carregado somente na primeira geração explícita."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from .config import DEFAULT_BASE_MODEL, ADAPTER_PATH


class ModelUnavailableError(RuntimeError):
    pass


def has_real_adapter(path: Path) -> bool:
    path = Path(path)
    return (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("adapter_model.safetensors", "adapter_model.bin"))


class FineTunedMedicalLLM:
    def __init__(self, base_model=DEFAULT_BASE_MODEL, adapter_path=ADAPTER_PATH):
        self.base_model = base_model
        # None is only for explicit base-model evaluation, never an automatic fallback.
        self.adapter_path = Path(adapter_path) if adapter_path is not None else None
        self.model = None
        self.tokenizer = None
        self._lock = RLock()

    def load(self):
        with self._lock:
            if self.model is not None:
                return self
            if self.adapter_path is not None:
                if not has_real_adapter(self.adapter_path):
                    raise ModelUnavailableError("Adapter real não encontrado. Execute train_finetune --real em GPU/Colab; o mock não contém pesos.")
                try:
                    config = json.loads((self.adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
                    adapter_base = config.get("base_model_name_or_path")
                    if adapter_base and adapter_base != self.base_model:
                        raise ModelUnavailableError(f"Adapter requer modelo base {adapter_base}; recebido {self.base_model}.")
                except (OSError, ValueError) as exc:
                    raise ModelUnavailableError("Configuração do adapter inválida.") from exc
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                if self.adapter_path is not None:
                    from peft import PeftModel
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer_source = self.adapter_path if self.adapter_path is not None and (self.adapter_path / "tokenizer_config.json").exists() else self.base_model
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
                if self.adapter_path is not None:
                    model = PeftModel.from_pretrained(model, str(self.adapter_path))
                model = model.to(device).eval()
                self.tokenizer, self.model = tokenizer, model
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise ModelUnavailableError("Não foi possível carregar a LLM local. Verifique requirements-finetuning.txt, pesos e memória CPU/GPU.") from exc
            return self

    def generate(self, prompt: str, max_new_tokens=256) -> str:
        return self.generate_messages([{"role": "user", "content": prompt}], max_new_tokens)

    def generate_messages(self, messages, max_new_tokens=256) -> str:
        with self._lock:
            self.load()
            import torch
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            # Do not silently truncate patient facts, sources, or safety instructions.
            capacity = getattr(self.model.config, "max_position_embeddings", 32768)
            if inputs["input_ids"].shape[-1] + max_new_tokens > capacity:
                raise ModelUnavailableError("Contexto excede a capacidade do modelo; reduza o contexto fornecido.")
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                             pad_token_id=self.tokenizer.pad_token_id)
            return self.tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
