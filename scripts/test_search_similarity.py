import os
import sys

# Ensure project root is on sys.path when run from the scripts folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.text_encoder import encode_text
from src.search import _search_by_embedding


if __name__ == '__main__':
    query = 'dog'
    print('Encoding query:', query)
    q_emb = encode_text(query)
    results = _search_by_embedding(q_emb, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['relative_path']} - similarity={r['similarity']:.4f} score={r['score']:.2f}")
