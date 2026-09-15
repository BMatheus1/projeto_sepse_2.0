import builtins
import json
from unittest.mock import Mock
import pytest

from src.tc_fase3 import langchain_pipeline as lp
from src.tc_fase3.fine_tuned_llm import FineTunedMedicalLLM, ModelUnavailableError
from src.tc_fase3.protocol_retriever import ProtocolRetriever


@pytest.fixture
def context():
    return dict(question="Quais exames estão pendentes?", patient={"patient_id": "SINTETICO_TESTE"},
                risk={"risk_level": "alto", "used_model": "regra"}, pending_exams=["hemocultura"],
                protocols=[{"source": "protocolo_teste.md", "content": "Dado ausente não deve ser inventado."}],
                sources=["protocolo_teste.md", "pacientes_sinteticos.jsonl"])


def fake_model():
    model = Mock(spec=["generate_messages"])
    model.generate_messages.return_value = "Dados do paciente: exame pendente. Inferência: aguardar revisão da equipe."
    return model


def test_real_lcel_pipeline_receives_all_context(context):
    from langchain_core.runnables import RunnableSequence
    model = fake_model()
    pipeline = lp.create_pipeline(model)
    assert isinstance(pipeline, RunnableSequence)
    assert pipeline.invoke(context) == model.generate_messages.return_value
    messages = model.generate_messages.call_args.args[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "não" in messages[0]["content"].lower()
    for value in ["SINTETICO_TESTE", "hemocultura", "alto", "regra", "protocolo_teste.md", context["question"], "Dado ausente"]:
        assert value in messages[1]["content"]


def test_no_adapter_is_fallback_without_loading(context, tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "get_model", Mock(side_effect=AssertionError("Must not load weights")))
    result = lp.generate_contextual_answer(**context, adapter_path=tmp_path, fallback=lambda: "seguro")
    assert result["generation_mode"] == "template_fallback"
    assert result["generation_fallback_reason"] == "no_real_adapter"
    assert result["answer"] == "seguro"


def test_success_reports_custom_llm_mode(context):
    result = lp.generate_contextual_answer(**context, model=fake_model(), fallback=lambda: "fallback")
    assert result["generation_mode"] == "fine_tuned_langchain"
    assert "validação humana" in result["answer"]
    assert "protocolo_teste.md" in result["answer"]


@pytest.mark.parametrize("text", ["", "Administre 500 mg agora.", "O paciente tem sepse."])
def test_rejects_invalid_generated_content(context, text):
    model = fake_model()
    model.generate_messages.return_value = text
    assert lp.generate_contextual_answer(**context, model=model, fallback=lambda: "seguro")["answer"] == "seguro"


def test_generation_failure_and_blocked_question(context):
    model = fake_model()
    model.generate_messages.side_effect = RuntimeError("OOM")
    result = lp.generate_contextual_answer(**context, model=model, fallback=lambda: "seguro")
    assert result["generation_mode"] == "template_fallback"
    model.reset_mock()
    result = lp.generate_contextual_answer(**context, model=model, blocked=True, fallback=lambda: "recusa")
    model.generate_messages.assert_not_called()
    assert result["answer"] == "recusa"


def test_missing_langchain_dependency_falls_back(context, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("langchain_core"):
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert lp.generate_contextual_answer(**context, model=fake_model(), fallback=lambda: "seguro")["generation_mode"] == "template_fallback"
    assert ProtocolRetriever().retrieve("sepse")


def test_retriever_runnable_returns_documents():
    from langchain_core.documents import Document
    retriever = ProtocolRetriever()
    docs = retriever.as_runnable(limit=2).invoke("sepse exames")
    assert len(docs) == 2
    assert all(isinstance(doc, Document) and doc.metadata["source"] for doc in docs)


def test_missing_adapter_fails_before_heavy_import(tmp_path):
    with pytest.raises(ModelUnavailableError, match="Adapter real não encontrado"):
        FineTunedMedicalLLM(adapter_path=tmp_path).load()


def test_assistant_graph_and_api_use_real_pipeline(monkeypatch, tmp_path):
    from src.tc_fase3.assistant import SepsisAssistant
    from src.tc_fase3.langgraph_flow import _run_langgraph
    from src.tc_fase3.schemas import AssistantResponse
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fake-base"}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"test-only")
    monkeypatch.setenv("FASE3_ADAPTER_PATH", str(tmp_path))
    model = fake_model()
    factory = Mock(return_value=model)
    monkeypatch.setattr(lp, "get_model", factory)
    direct = SepsisAssistant().answer("PACIENTE_001", "Quais exames estão pendentes?")
    graph = _run_langgraph(dict(patient_id="PACIENTE_001", question="Quais exames estão pendentes?", executed_nodes=[]))
    for result in (direct, graph):
        assert result["generation_mode"] == "fine_tuned_langchain"
        assert AssistantResponse(**result).generation_mode == "fine_tuned_langchain"
        assert "audit" in result["executed_nodes"][-1]
    assert model.generate_messages.call_count == 2
    assert factory.call_args.args[0] == "fake-base"


@pytest.mark.parametrize("cuda", [False, True])
def test_inference_loads_peft_and_decodes_only_new_tokens(cuda, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from contextlib import nullcontext
    from unittest.mock import MagicMock
    (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": "test-base"}')
    (tmp_path / "adapter_model.safetensors").write_bytes(b"test-only")
    class Encoding(dict):
        def to(self, device):
            return self
    encoding = Encoding(input_ids=SimpleNamespace(shape=(1, 3)))
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.return_value = encoding
    tokenizer.decode.return_value = "resposta gerada"
    tokenizer.apply_chat_template.return_value = "rendered chat"
    base = Mock()
    adapted = Mock()
    adapted.to.return_value = adapted
    adapted.eval.return_value = adapted
    adapted.config.max_position_embeddings = 1024
    adapted.device = "cuda" if cuda else "cpu"
    class OutputTensor:
        def __getitem__(self, key):
            row, token_slice = key
            assert row == 0
            return [10, 11, 12, 20, 21][token_slice]
    adapted.generate.return_value = OutputTensor()
    auto_model = Mock(from_pretrained=Mock(return_value=base))
    peft = Mock(from_pretrained=Mock(return_value=adapted))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda), float16="fp16", float32="fp32", inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Mock(from_pretrained=Mock(return_value=tokenizer)), AutoModelForCausalLM=auto_model))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=peft))
    llm = FineTunedMedicalLLM("test-base", tmp_path)
    assert llm.generate("pergunta", max_new_tokens=10) == "resposta gerada"
    peft.from_pretrained.assert_called_once_with(base, str(tmp_path))
    adapted.to.assert_called_once_with("cuda" if cuda else "cpu")
    tokenizer.decode.assert_called_once_with([20, 21], skip_special_tokens=True)
    llm.load()
    auto_model.from_pretrained.assert_called_once()
