#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.search import image_to_image_search

img = 'dataset/images/scoiattolo_24317.jpg'
print('Running image-to-image search for', img)
results = image_to_image_search(img, top_k=5)
for i, r in enumerate(results, 1):
    print(f"{i}. {r['relative_path']}  score={r['score']:.2f}% sim={r['similarity']:.4f}")
