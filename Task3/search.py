#!/usr/bin/env python3

import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def initialize_vector_database():
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    database = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

    return database

def execute_search_tests(database):
    test_scenarios = [
        {
            "query": "Who is Lord of Air?",
        },
        {
            "query": "What is the biggest war in history?",
        },
        {
            "query": "What was the role of the first elementals?",
        },
    ]

    for index, scenario in enumerate(test_scenarios, 1):
        print(f"Query: '{scenario['query']}'\n")

        search_results = database.similarity_search(scenario['query'], k=3)

        for result_index, result in enumerate(search_results, 1):
            print(f"   {result_index}. Source: {result.metadata['source']}")
            print(f"      Category: {result.metadata['category']}")
            print(f"      Chunk ID: {result.metadata['chunk_id']}")
            print(f"      Chunk Size: {len(result.page_content)} characters")
            print(f"      Position: {result.metadata.get('chunk_index', 'N/A')} of {result.metadata.get('total_chunks', 'N/A')}")
            print(f"      Content:")
            print(f"      {result.page_content}")
            print()

def run():
    try:
        print("Loading vector database")
        vector_database = initialize_vector_database()
        print("Vector database loaded successfully")
        execute_search_tests(vector_database)

        print(f"Documents in index: {vector_database._collection.count()}")

    except Exception as error:
        print(f"{error}")

if __name__ == "__main__":
    run()