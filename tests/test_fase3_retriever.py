from src.tc_fase3.protocol_retriever import ProtocolRetriever


def test_retriever_finds_sepsis_protocol():
    results = ProtocolRetriever().search("sepse triagem lactato hipotensão")
    assert results


def test_retriever_returns_source():
    result = ProtocolRetriever().search("limites do assistente")
    assert "source" in result[0]
    assert result[0]["source"].endswith(".md")
