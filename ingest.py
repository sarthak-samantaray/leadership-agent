#!/usr/bin/env python3
"""Build the Chroma vector store from PDFs in data/documents/."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

# Before HF/Chroma: avoids extra tokenizer subprocesses; pairs with multiprocess>=0.70.19 for Py3.12.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.runtime_bootstrap import apply_runtime_bootstrap

apply_runtime_bootstrap()

from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import load_pdfs_from_directory
from src.utils.config import LeadershipAgentConfig
from src.utils.logger import get_logger
from src.vectorstore.store_manager import build_vectorstore

log = get_logger(__name__)


def main() -> None:
    cfg = LeadershipAgentConfig.load(ROOT / "leadership_agent_config.json")
    doc_dir = cfg.resolve(cfg.paths.documents_dir)
    raw = load_pdfs_from_directory(doc_dir)
    if not raw:
        log.error("No PDFs found in %s — add files and retry.", doc_dir)
        sys.exit(1)
    chunks = chunk_documents(raw, cfg)
    log.info("Indexing %s chunks …", len(chunks))
    vs = build_vectorstore(cfg, chunks)
    del vs
    gc.collect()
    log.info("Done. Vector store at %s", cfg.resolve(cfg.paths.vectorstore_dir))


if __name__ == "__main__":
    main()
