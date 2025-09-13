#!/usr/bin/env python3

import json
import os
import time
from pathlib import Path
from typing import List, Set
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeIndexUpdater:
    def __init__(self, data_source_path: str = "./knowledge_base", storage_path: str = "./chroma_db"):
        self.data_source_path = Path(data_source_path)
        self.storage_path = storage_path
        self.embedding_model_name = "all-mpnet-base-v2"
        self.embedding_engine = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.chunk_processor = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )

        self.source_documents = []
        self.processed_chunks = []
        self.metadata_file = Path(self.storage_path) / "document_metadata.json"

    def retrieve_documents(self) -> List[Document]:
        self.source_documents = []
        for file_path in self.data_source_path.rglob("*.txt"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    if content:
                        relative_path = file_path.relative_to(self.data_source_path)
                        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"

                        metadata = {
                            "source": str(relative_path),
                            "category": category,
                            "filename": file_path.name,
                            "file_path": str(file_path),
                            "file_size": len(content),
                            "last_modified": file_path.stat().st_mtime
                        }

                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        self.source_documents.append(doc)

                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        logger.info(f"Retrieved: {len(self.source_documents)} documents")
        return self.source_documents

    def load_existing_metadata(self) -> dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_metadata(self, metadata: dict):
        os.makedirs(self.storage_path, exist_ok=True)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def identify_changed_documents(self) -> tuple[List[Document], Set[str]]:
        existing_metadata = self.load_existing_metadata()
        changed_docs = []
        changed_sources = set()
        
        for doc in self.source_documents:
            source = doc.metadata["source"]
            last_modified = doc.metadata["last_modified"]
            
            if (source not in existing_metadata or 
                existing_metadata[source]["last_modified"] != last_modified):
                changed_docs.append(doc)
                changed_sources.add(source)
                
        return changed_docs, changed_sources

    def segment_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            segments = self.chunk_processor.split_documents([doc])
            valid_segments = []
            for i, segment in enumerate(segments):
                if len(segment.page_content.strip()) >= 10:
                    segment.metadata.update({
                        "chunk_id": f"{doc.metadata['filename']}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(segments)
                    })
                    valid_segments.append(segment)
            chunks.extend(valid_segments)
        return chunks

    def update_vector_database(self):
        os.makedirs(self.storage_path, exist_ok=True)
        
        try:
            database = Chroma(
                persist_directory=self.storage_path,
                embedding_function=self.embedding_engine
            )
        except:
            database = Chroma.from_documents(
                documents=[],
                embedding=self.embedding_engine,
                persist_directory=self.storage_path
            )

        self.retrieve_documents()
        changed_docs, changed_sources = self.identify_changed_documents()
        
        if not changed_docs:
            logger.info("No changes detected in documents")
            return database

        logger.info(f"Processing {len(changed_docs)} changed documents")
        
        new_chunks = self.segment_documents(changed_docs)
        
        if new_chunks:
            ids_to_delete = []
            for source in changed_sources:
                docs_to_delete = database.get(where={"source": source})
                ids_to_delete.extend(docs_to_delete["ids"])
                
            if ids_to_delete:
                database.delete(ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} old chunks")
            
            database.add_documents(new_chunks)
            logger.info(f"Added {len(new_chunks)} new chunks")

        new_metadata = {
            doc.metadata["source"]: {
                "last_modified": doc.metadata["last_modified"]
            }
            for doc in self.source_documents
        }
        self.save_metadata(new_metadata)

        return database

    def execute_queries(self, database: Chroma, sample_queries: List[str] = None):
        if sample_queries is None:
            sample_queries = [
                "Who is Lord of Air?",
                "Who is enraged Fire Elemental?",
                "Who is fallen titan who turned to demon?",
            ]

        for query in sample_queries:
            logger.info(f"\nQuery: {query}")
            matches = database.similarity_search(query, k=3)

            for i, match in enumerate(matches, 1):
                logger.info(f"  {i}. {match.metadata['source']}")
                logger.info(f"     {match.page_content[:150]}...")

    def record_metrics(self, database: Chroma, execution_duration: float, chunks_added: int):
        try:
            collection = database._collection
            total_chunks = collection.count()
        except:
            total_chunks = len(database.get()["ids"])

        metrics = {
            "model": {
                "name": self.embedding_model_name,
                "repository": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
                "embedding_size": 768
            },
            "knowledge_base": {
                "path": str(self.data_source_path),
                "total_documents": len(self.source_documents),
                "processed_chunks": chunks_added,
                "total_chunks_in_db": total_chunks
            },
            "processing": {
                "time_seconds": execution_duration,
                "time_minutes": execution_duration / 60,
                "chunks_per_second": chunks_added / execution_duration if execution_duration > 0 else 0
            },
            "vectorstore": {
                "type": "ChromaDB",
                "persist_directory": self.storage_path
            }
        }

        with open("update_stats.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

def run():
    logger.info("Initializing Knowledge Base Update")
    
    start_timestamp = time.time()

    updater = KnowledgeIndexUpdater()
    database = updater.update_vector_database()
    
    chunks_added = len(database.get()["ids"]) if database else 0

    updater.execute_queries(database)

    elapsed_time = time.time() - start_timestamp
    performance_metrics = updater.record_metrics(database, elapsed_time, chunks_added)

    logger.info("\n" + "=" * 50)
    logger.info("Update Complete:")
    logger.info(f"  Model: {performance_metrics['model']['name']}")
    logger.info(f"  Embedding Dimension: {performance_metrics['model']['embedding_size']}")
    logger.info(f"  Documents Processed: {performance_metrics['knowledge_base']['total_documents']}")
    logger.info(f"  Chunks Added: {performance_metrics['knowledge_base']['processed_chunks']}")
    logger.info(f"  Total Chunks: {performance_metrics['knowledge_base']['total_chunks_in_db']}")
    logger.info(f"  Processing Time: {performance_metrics['processing']['time_minutes']:.2f} minutes")
    logger.info("=" * 50)

if __name__ == "__main__":
    run()