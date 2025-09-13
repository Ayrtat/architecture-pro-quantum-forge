#!/usr/bin/env python3

import os
import csv
from datetime import datetime
from app import RAGEngine, RAGSettings

class GoldenTestSuite:
    def __init__(self, questions_file='golden_questions.txt'):
        self.questions = self._load_questions(questions_file)
        self.rag_engine = RAGEngine(RAGSettings())
        self.results_file = f'golden_test_results/{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    def _load_questions(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _evaluate_response(self, query, response_data):
        num_sources = response_data.get('num_sources', 0)
        response_text = response_data.get('response', '')
        
        is_found = num_sources > 0
        completeness = self._assess_completeness(response_text)
        
        return {
            'query': query,
            'is_found': is_found,
            'num_sources': num_sources,
            'completeness': completeness,
            'response': response_text
        }

    def _assess_completeness(self, response):
        if len(response) < 50:
            return 'low'
        elif len(response) < 200:
            return 'medium'
        else:
            return 'high'

    def run_test_suite(self):
        results = []
        
        for query in self.questions:
            try:
                response_data = self.rag_engine.create_response(query)
                evaluation = self._evaluate_response(query, response_data)
                results.append(evaluation)
                print(f"Query: {query}")
                print(f"Found: {evaluation['is_found']}, Completeness: {evaluation['completeness']}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
        
        self._save_results(results)
        self._generate_summary(results)

    def _save_results(self, results):
        with open(self.results_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['query', 'is_found', 'num_sources', 'completeness', 'response']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    def _generate_summary(self, results):
        total_queries = len(results)
        found_queries = sum(1 for r in results if r['is_found'])
        
        print("\n--- Test Suite Summary ---")
        print(f"Total Queries: {total_queries}")
        print(f"Queries with Sources: {found_queries} ({found_queries/total_queries*100:.2f}%)")
        
        completeness_counts = {}
        for result in results:
            completeness_counts[result['completeness']] = completeness_counts.get(result['completeness'], 0) + 1
        
        print("\nCompleteness Distribution:")
        for level, count in completeness_counts.items():
            print(f"{level.capitalize()}: {count} ({count/total_queries*100:.2f}%)")

def main():
    test_suite = GoldenTestSuite()
    test_suite.run_test_suite()

if __name__ == "__main__":
    main()
