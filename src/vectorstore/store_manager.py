from pathlib import Path
import time
import threading
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import LeadershipAgentConfig
from src.utils.logger import get_logger

log = get_logger(__name__)
_VS_LOCK = threading.Lock()
_VS_CACHE: dict[str, Chroma] = {}


def _make_embeddings(cfg: LeadershipAgentConfig) -> HuggingFaceEmbeddings:
    log.info("Loading HuggingFace embeddings: %s", cfg.embeddings.model_name)
    return HuggingFaceEmbeddings(
        model_name=cfg.embeddings.model_name,
        model_kwargs={"device": cfg.embeddings.device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(
    cfg: LeadershipAgentConfig,
    documents: List[Document],
    persist_dir: Optional[Path] = None,
) -> Chroma:
    persist = persist_dir or cfg.resolve(cfg.paths.vectorstore_dir)
    persist.mkdir(parents=True, exist_ok=True)
    emb = _make_embeddings(cfg)
    if documents:
        vs = Chroma.from_documents(
            documents=documents,
            embedding=emb,
            persist_directory=str(persist),
        )
    else:
        vs = Chroma(
        embedding_function=emb,
        persist_directory=str(persist),
        )
    with _VS_LOCK:
        _VS_CACHE[str(persist.resolve())] = vs
    return vs


def load_vectorstore(cfg: LeadershipAgentConfig, persist_dir: Optional[Path] = None) -> Chroma:
    persist = persist_dir or cfg.resolve(cfg.paths.vectorstore_dir)
    if not persist.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist}. Run: python ingest.py"
        )
    key = str(persist.resolve())
    with _VS_LOCK:
        cached = _VS_CACHE.get(key)
        if cached is not None:
            return cached

    # Chroma shared client creation can race under parallel tool calls.
    # Serialize first creation and add a short retry to avoid transient KeyError.
    with _VS_LOCK:
        cached = _VS_CACHE.get(key)
        if cached is not None:
            return cached
        emb = _make_embeddings(cfg)
        try:
            vs = Chroma(
                embedding_function=emb,
                persist_directory=str(persist),
            )
        except KeyError:
            time.sleep(0.15)
            vs = Chroma(
                embedding_function=emb,
                persist_directory=str(persist),
            )
        _VS_CACHE[key] = vs
        return vs
