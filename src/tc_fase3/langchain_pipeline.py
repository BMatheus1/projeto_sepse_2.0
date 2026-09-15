"""LCEL real: contexto/Document -> ChatPromptTemplate -> PEFT local -> parser."""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from .config import ADAPTER_PATH, DEFAULT_BASE_MODEL
from .fine_tuned_llm import FineTunedMedicalLLM, has_real_adapter
from .prompts import create_chat_prompt
from .safety import complete_safe_answer, generated_answer_is_unsafe

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model(base_model: str, adapter_path: str):
    return FineTunedMedicalLLM(base_model, adapter_path)


def context_values(context):
    from langchain_core.documents import Document
    documents = [Document(page_content=p["content"], metadata={"source": p["source"]})
                 for p in context["protocols"]]
    return {
        "question": context["question"],
        "patient_context": json.dumps(context["patient"], ensure_ascii=False),
        "pending_exams": json.dumps(context["pending_exams"], ensure_ascii=False),
        "risk_context": json.dumps(context["risk"], ensure_ascii=False),
        "protocol_context": "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in documents),
        "sources": ", ".join(context["sources"]),
    }


def create_pipeline(model):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda

    def invoke_model(prompt):
        roles = {"system": "system", "human": "user", "ai": "assistant"}
        messages = [{"role": roles[m.type], "content": m.content} for m in prompt.to_messages()]
        return model.generate_messages(messages)

    return (RunnableLambda(context_values) | create_chat_prompt()
            | RunnableLambda(invoke_model) | StrOutputParser())


def generate_contextual_answer(*, question, patient, risk, pending_exams,
                               protocols, sources, fallback, blocked=False,
                               model=None, adapter_path=None):
    """Fallback preserves local behavior; failures are logged without patient content."""
    reason = None
    path = Path(adapter_path or os.environ.get("FASE3_ADAPTER_PATH", str(ADAPTER_PATH)))
    if blocked:
        reason = "blocked_question"
    elif model is None and not has_real_adapter(path):
        reason = "no_real_adapter"
    if reason is None:
        try:
            if model is None:
                adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
                base = os.environ.get("FASE3_BASE_MODEL") or adapter_config.get("base_model_name_or_path") or DEFAULT_BASE_MODEL
                model = get_model(base, str(path.resolve()))
            answer = create_pipeline(model).invoke(dict(
                question=question, patient=patient, risk=risk, pending_exams=pending_exams,
                protocols=protocols, sources=sources))
            if not answer.strip() or generated_answer_is_unsafe(answer):
                raise ValueError("Geração vazia ou reprovada pela validação de segurança.")
            answer += "\n\nEsta resposta não é diagnóstico definitivo, não prescreve e exige validação humana obrigatória."
            answer += "\nFontes do contexto: " + ", ".join(sources) + "."
            return {"answer": complete_safe_answer(answer, sources), "generation_mode": "fine_tuned_langchain",
                    "generation_fallback_reason": None}
        except Exception as exc:
            reason = "generation_failed:" + type(exc).__name__
            logger.warning("LLM local indisponível ou geração rejeitada (%s)", type(exc).__name__)
    return {"answer": fallback(), "generation_mode": "template_fallback", "generation_fallback_reason": reason}
