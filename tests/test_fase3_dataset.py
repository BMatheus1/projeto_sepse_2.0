import json

from src.tc_fase3.anonymization import anonymize_text
from src.tc_fase3.config import FINE_TUNING_DATASET_PATH
from src.tc_fase3.prepare_finetuning_dataset import prepare_dataset


def test_fase3_dataset_is_generated():
    summary = prepare_dataset()
    assert summary["valid_records"] >= 1
    assert FINE_TUNING_DATASET_PATH.exists()


def test_fase3_dataset_records_have_messages():
    prepare_dataset()
    first_line = FINE_TUNING_DATASET_PATH.read_text(encoding="utf-8-sig").splitlines()[0]
    record = json.loads(first_line)
    assert "messages" in record
    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]


def test_anonymization_removes_email_cpf_phone_and_simulated_name():
    text = "Nome: João Silva, CPF 123.456.789-10, email joao@email.com, telefone (11) 98888-7777"
    anonymized = anonymize_text(text)
    assert "joao@email.com" not in anonymized
    assert "123.456.789-10" not in anonymized
    assert "98888-7777" not in anonymized
    assert "João Silva" not in anonymized
    assert "[EMAIL_REMOVIDO]" in anonymized
    assert "[CPF_REMOVIDO]" in anonymized
    assert "[TELEFONE_REMOVIDO]" in anonymized
    assert "[NOME_REMOVIDO]" in anonymized
