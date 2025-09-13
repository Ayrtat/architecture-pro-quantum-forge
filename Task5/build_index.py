#!/usr/bin/env python3

import json
import os
import time
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class KnowledgeIndexBuilder:
    def __init__(self, data_source_path: str = "./knowledge_base"):
        self.data_source_path = Path(data_source_path)
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

    def retrieve_documents(self) -> List[Document]:
        for file_path in self.data_source_path.rglob("*.txt"):
            if file_path.is_file():
                print(file_path)
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
                            "file_size": len(content)
                        }

                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        self.source_documents.append(doc)

                except Exception as e:
                    print(e)

        print(f"Retrieved: {len(self.source_documents)} documents")
        return self.source_documents

    def segment_documents(self) -> List[Document]:
        for doc in self.source_documents:
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

            self.processed_chunks.extend(valid_segments)

        return self.processed_chunks

    def build_vector_database(self, storage_path: str = "./chroma_db") -> Chroma:
        os.makedirs(storage_path, exist_ok=True)

        database = Chroma.from_documents(
            documents=self.processed_chunks,
            embedding=self.embedding_engine,
            persist_directory=storage_path
        )

        return database

    def execute_queries(self, database: Chroma, sample_queries: List[str] = None):
        if sample_queries is None:
            sample_queries = [
                "Who is Lord of Air?",
                "Who is enraged Fire Elemental?",
                "Who is fallen titan who turned to demon?",
            ]

        for query in sample_queries:
            print(f"\nQuery: {query}")
            matches = database.similarity_search(query, k=3)

            for i, match in enumerate(matches, 1):
                print(f"  {i}. {match.metadata['source']}")
                print(f"     {match.page_content[:150]}...")
                print()

    def record_metrics(self, database: Chroma, execution_duration: float):
        metrics = {
            "model": {
                "name": self.embedding_model_name,
                "repository": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2  ",
                "embedding_size": 768
            },
            "knowledge_base": {
                "path": str(self.data_source_path),
                "total_documents": len(self.source_documents),
                "total_chunks": len(self.processed_chunks)
            },
            "processing": {
                "time_seconds": execution_duration,
                "time_minutes": execution_duration / 60,
                "chunks_per_second": len(self.processed_chunks) / execution_duration
            },
            "vectorstore": {
                "type": "ChromaDB",
                "persist_directory": "./chroma_db"
            }
        }

        with open("indexing_stats.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return metrics

def run():
    print("Initializing\n")

    start_timestamp = time.time()

    builder = KnowledgeIndexBuilder()
    builder.retrieve_documents()
    builder.segment_documents()

    vector_database = builder.build_vector_database()

    builder.execute_queries(vector_database)

    elapsed_time = time.time() - start_timestamp
    performance_metrics = builder.record_metrics(vector_database, elapsed_time)

    print("\n" + "=" * 50)
    print("Final Metrics:")
    print(f"  Model: {performance_metrics['model']['name']}")
    print(f"  Embedding Dimension: {performance_metrics['model']['embedding_size']}")
    print(f"  Documents Processed: {performance_metrics['knowledge_base']['total_documents']}")
    print(f"  Chunks Created: {performance_metrics['knowledge_base']['total_chunks']}")
    print(f"  Processing Time: {performance_metrics['processing']['time_minutes']:.2f} minutes")
    print(f"  Processing Rate: {performance_metrics['processing']['chunks_per_second']:.1f} chunks/sec")
    print("=" * 50)

if __name__ == "__main__":
    run()