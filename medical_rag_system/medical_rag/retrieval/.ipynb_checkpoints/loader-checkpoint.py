"""LocalRAGLoader: Load RAG library with passage-level chunking."""
import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ..utils.logging import get_logger
from .utils import chunk_text, normalize_embeddings, clean_text

logger = get_logger(__name__)

class LocalRAGLoader:
    def __init__(
        self,
        library_dir: str,
        library_name: str,
        embedding_model_name: str,
        chunk_size: int = 300,
        chunk_overlap: int = 150,
    ):
        self.library_dir = Path(library_dir)
        self.library_name = library_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        logger.info("loading_rag_library", library_dir=str(library_dir), library_name=library_name)

        metadata_file = self.library_dir / f"{library_name}_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        logger.info("loading_embedding_model", model=embedding_model_name)
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.index, self.documents = self._load_or_create_chunked_index()

        logger.info(
            "rag_library_loaded",
            num_documents=len(self.documents),
            index_size=self.index.ntotal if self.index else 0
        )

    def _load_or_create_chunked_index(self) -> tuple:
        chunked_index_file = self.library_dir / f"{self.library_name}_chunked_index.faiss"
        chunked_docs_file = self.library_dir / f"{self.library_name}_chunked_documents.pkl"

        if chunked_index_file.exists() and chunked_docs_file.exists():
            logger.info("loading_existing_chunked_index")
            index = faiss.read_index(str(chunked_index_file))
            with open(chunked_docs_file, 'rb') as f:
                documents = pickle.load(f)
            return index, documents

        logger.info("creating_chunked_index_from_documents")
        docs_file = self.library_dir / f"{self.library_name}_documents.pkl"
        with open(docs_file, 'rb') as f:
            original_docs = pickle.load(f)

        chunked_docs = []
        all_chunks = []

        for doc_id, doc in enumerate(original_docs):
            text = clean_text(doc.get("text", ""))
            source = doc.get("source", "unknown")

            chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

            for chunk_idx, (chunk_text_piece, offset) in enumerate(chunks):
                chunked_doc = {
                    "text": chunk_text_piece,
                    "doc_id": f"{doc_id}_{chunk_idx}",
                    "parent_doc_id": doc_id,
                    "source": source,
                    "offset": offset,
                    "source_url": doc.get("url", None)
                }
                chunked_docs.append(chunked_doc)
                all_chunks.append(chunk_text_piece)

        logger.info("generating_chunk_embeddings", num_chunks=len(all_chunks))
        embeddings = self.embedding_model.encode(
            all_chunks, normalize_embeddings=True, show_progress_bar=True, batch_size=32
        ).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        faiss.write_index(index, str(chunked_index_file))
        with open(chunked_docs_file, 'wb') as f:
            pickle.dump(chunked_docs, f)

        logger.info("chunked_index_created_and_saved", num_passages=len(chunked_docs))

        return index, chunked_docs
