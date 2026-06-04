#!/usr/bin/env python
"""Quick test to verify similarity scores are correct."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.search import text_to_image_search

if __name__ == '__main__':
    print("Testing search with query: 'dog'")
    print("-" * 70)
    
    results = text_to_image_search('dog', top_k=5)
    
    if not results:
        print("ERROR: No results returned!")
        sys.exit(1)
    
    for i, r in enumerate(results, 1):
        score = r['score']
        similarity = r['similarity']
        path = r['relative_path']
        print(f"{i}. {path}")
        print(f"   Score: {score:.2f}% | Similarity: {similarity:.4f}")
        
        # Verify score is correct
        expected_score = similarity * 100.0
        if abs(score - expected_score) > 0.01:
            print(f"   ERROR: Score mismatch! Got {score}, expected {expected_score}")
            sys.exit(1)
    
    print("-" * 70)
    print("✓ All scores are correct!")
    print(f"✓ Scores range: {results[-1]['score']:.2f}% to {results[0]['score']:.2f}%")
