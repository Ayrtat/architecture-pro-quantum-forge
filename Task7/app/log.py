import csv
import os
from datetime import datetime
from typing import List, Dict, Any

def log_request_to_csv(
    query: str,
    response_data: Dict[str, Any],
    csv_file_path: str = "request_logs.csv"
):
    timestamp = datetime.now().isoformat()
    chunks_found = response_data.get("num_sources", 0) > 0
    response_length = len(response_data.get("response", ""))
    
    response_text = response_data.get("response", "").lower()
    keywords = ["unknown", "no data", "not found", "sorry", "error", "not provided"]
    is_successful = (
        response_length > 50 and
        not any(keyword in response_text for keyword in keywords)
    )
    
    sources = response_data.get("sources", [])
    source_names = "; ".join([src.get("source", "Unknown") for src in sources])

    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "chunks_found": chunks_found,
        "response_length": response_length,
        "is_successful": is_successful,
        "sources": source_names
    }

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["timestamp", "query", "chunks_found", "response_length", "is_successful", "sources"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)