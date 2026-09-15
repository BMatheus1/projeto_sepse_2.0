from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .config import PROTOCOLS_DIR


@dataclass
class ProtocolDocument:
    source: str
    content: str


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-zÀ-ÿ0-9_]+", text) if len(token) >= 3}


class ProtocolRetriever:
    def __init__(self, protocols_dir: Path = PROTOCOLS_DIR) -> None:
        self.protocols_dir = protocols_dir
        self.documents = self._load_documents()

    def _load_documents(self) -> List[ProtocolDocument]:
        if not self.protocols_dir.exists():
            return []
        return [
            ProtocolDocument(source=path.name, content=path.read_text(encoding="utf-8-sig"))
            for path in sorted(self.protocols_dir.glob("*.md"))
        ]

    def count(self) -> int:
        return len(self.documents)

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        scored: List[Dict[str, Any]] = []
        for document in self.documents:
            doc_tokens = tokenize(document.content)
            score = len(query_tokens & doc_tokens)
            if score == 0 and any(term in document.content.lower() for term in ["sepse", "assistente", "validação"]):
                score = 1
            if score > 0:
                scored.append({"source": document.source, "content": document.content[:1200], "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def as_langchain_documents(self) -> List[Any]:
        try:
            from langchain_core.documents import Document
        except ImportError:
            return []
        return [Document(page_content=doc.content, metadata={"source": doc.source}) for doc in self.documents]

    def as_runnable(self, limit: int = 3):
        """Retriever lexical local consumível via invoke e composição LCEL."""
        from langchain_core.documents import Document
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda query: [
            Document(page_content=item["content"], metadata={"source": item["source"], "score": item["score"]})
            for item in self.search(query, limit=limit)
        ])

    def retrieve(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        try:
            runnable = self.as_runnable(limit)
        except ImportError:
            return self.search(query, limit)
        return [{"source": doc.metadata["source"], "score": doc.metadata["score"], "content": doc.page_content}
                for doc in runnable.invoke(query)]
