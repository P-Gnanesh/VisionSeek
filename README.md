# VisionSeek

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-OpenCLIP-EE4C2C?logo=pytorch&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Similarity%20Search-0099E5)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-Vision%2FLanguage-2F4F4F)

## Table of Contents

- [Contributors](#contributors)
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Model Architecture](#model-architecture)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Contributors

- Mohammed Afshan — https://github.com/MohdAfshan
- P Gnanesh — https://github.com/P-Gnanesh
- Ibrahim Sharif — https://github.com/1brahimSharif
- Namitha R — https://github.com/namitha2526
  
## Project Overview

VisionSeek is a semantic image search application built with Flask, OpenCLIP, and FAISS. It allows users to search an image library using natural language, a reference image, or a combined multimodal query. The backend converts images and text into shared embeddings, performs cosine-similarity search over a precomputed FAISS index, and returns ranked visual matches with interactive filtering, favorites, export, and feedback workflows.

The project is designed to be practical for recruiter and hiring-manager review: it demonstrates modern computer vision inference, vector search, API design, session-backed UI state, and GPU-aware embedding generation.

## Features

- Text-to-image semantic search using CLIP/OpenCLIP embeddings.
- Image-to-image search from uploaded query images.
- Multimodal search that blends image and text embeddings into one query vector.
- Relevance feedback and Rocchio-style reranking through `/search/rerank`.
- Category filtering backed by `metadata.json`.
- Favorites management persisted in `favorites.json`.
- Batch image upload for admins with automatic index updates.
- JSON and ZIP export flows for selected results.
- File-system based session history for recent text queries.
- API key protection for `/api/v1/*` routes when `VISIONSEEK_API_KEY` is set.
- GPU-aware embedding generation when CUDA is available.
- A polished single-page web UI with responsive search and results rendering.

## Tech Stack

- Backend: Flask, Flask-Session, Werkzeug
- ML / Vision: OpenCLIP, PyTorch, PIL
- Vector search: FAISS, NumPy
- Storage: Local JSON, NPY, and FAISS files on disk; no external database
- Frontend: HTML, CSS, JavaScript, Tailwind CSS CDN
- Utilities and tooling: python-docx for report generation, standard library helpers, shell-based dataset scripts

## Project Structure

```text
MinProject/
|-- app.py                  # Flask application entry point and route handlers
|-- config.py               # Centralized paths, model name, and runtime limits
|-- favorites.json          # Persisted favorites list
|-- metadata.json           # Image filename to category mapping
|-- image_embeddings.npy    # Precomputed image embeddings
|-- image_paths.npy         # Paths aligned with the embedding array
|-- faiss_index.bin         # Persisted FAISS index
|-- generate_report.py      # Generates a Word report for the project
|-- flatten_raw_img_dataset.py
|-- quick_test.py
|-- tmp_image_test.py
|-- test_export_fix.html
|-- dataset/
|   `-- images/             # Flattened image dataset used by the app
|       `-- translate.py    # Dataset helper script
|-- docs/
|   `-- API.md              # REST API reference
|-- scripts/
|   |-- flatten_raw_img_dataset.py
|   `-- test_search_similarity.py
|-- src/
|   |-- api.py              # Versioned API blueprint
|   |-- faiss_utils.py      # FAISS index helpers
|   |-- favorites.py        # Favorites persistence helpers
|   |-- generate_embeddings.py
|   |-- image_encoder.py    # OpenCLIP image embeddings
|   |-- index_manager.py    # Add new images to the stored index
|   |-- metadata_store.py   # Category metadata loading/filtering
|   |-- multimodal_encoder.py
|   |-- rerank.py           # Rocchio-style relevance feedback
|   |-- search.py           # Search orchestration and filtering
|   |-- text_encoder.py     # OpenCLIP text embeddings
|   `-- clip_test.py
|-- templates/
|   `-- index.html          # Main UI template
|-- utils/
|   `-- helpers.py          # Logging, image loading, and shared helpers
`-- flask_session/          # File-system session storage
```

## Installation

1. Clone the repository.

```bash
git clone <repository-url>
cd MinProject
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the runtime dependencies.

The repository does not currently include a `requirements.txt` file. Install the packages inferred from the codebase:

```bash
pip install flask flask-session numpy pillow torch open-clip-torch faiss-cpu python-docx werkzeug
```

If you are using a GPU-enabled environment, install the FAISS and PyTorch builds that match your platform and CUDA version.

4. Prepare the dataset and index files.

- Ensure images are present under `dataset/images/`.
- The application expects `image_embeddings.npy`, `image_paths.npy`, and `faiss_index.bin` in the project root.
- If those files are missing, generate them with the embedding script described below.

## Configuration

The following settings are defined in code and/or environment variables:

- `SECRET_KEY`: Flask session secret. Defaults to `dev-secret-key` if not set.
- `VISIONSEEK_API_KEY`: Optional API key for `/api/v1/*` routes. When set, clients must send `X-API-Key` with the same value.
- `X-Admin-Token`: Required for `/admin/upload`. The current implementation expects the hardcoded token `admin-upload-token`.

Other runtime settings are defined in `config.py`:

- `TOP_K_RESULTS = 5`
- `MAX_UPLOAD_BYTES = 16 MB`
- `MODEL_NAME = "ViT-B-32"`
- `PRETRAINED_TAG = "openai"`

Persistent local files used by the app:

- `favorites.json` stores saved favorites.
- `metadata.json` stores filename-to-category mappings.
- `flask_session/` stores session files.
- `image_embeddings.npy`, `image_paths.npy`, and `faiss_index.bin` store the search index assets.

To Be Added:

- A `.env` example file for local configuration.
- Externalization of the admin upload token into environment configuration.
- A formal `requirements.txt` or lock file.

## Usage

### Start the web application

```bash
python app.py
```

By default, Flask starts the development server with `debug=True` on `http://127.0.0.1:5000`.

### Generate embeddings and FAISS index

Use the embedding pipeline after adding or updating images in `dataset/images/`:

```bash
python src/generate_embeddings.py
```

The script scans the image folder, encodes images in batches, saves `image_embeddings.npy` and `image_paths.npy`, and writes `faiss_index.bin`.

### Flatten a raw dataset

If you have a nested `dataset/images/raw-img/` layout, use the flattening helper:

```bash
python scripts/flatten_raw_img_dataset.py
```

### Add uploaded images to the index

The admin upload flow uses `/admin/upload` with `X-Admin-Token: admin-upload-token` and automatically adds new files to the embeddings and FAISS index.

### Interact with the app

- Enter a text query on the home page and submit it.
- Upload a reference image to run image-to-image search.
- Use the category pills to narrow results.
- Mark images as favorites to persist them locally.
- Use the export actions to download JSON or ZIP bundles of selected results.

## API Endpoints

The versioned API is mounted at `/api/v1`. When `VISIONSEEK_API_KEY` is configured, send `X-API-Key: <value>` with every request.

### REST API

| Method | Endpoint | Description | Sample Response |
| --- | --- | --- | --- |
| `POST` | `/api/v1/search/text` | Search by natural-language text query. | `{ "query": "a red car", "results": [{ "image_path": "...", "filename": "...", "score": 87.3 }], "elapsed_ms": 42.1 }` |
| `POST` | `/api/v1/search/image` | Search by uploaded image file. | `{ "query": "query.jpg", "results": [{ "image_path": "...", "filename": "...", "score": 91.4 }], "elapsed_ms": 58.8 }` |
| `GET` | `/api/v1/health` | Report service status and index size. | `{ "status": "ok", "index_size": 26179 }` |

### Web and application routes

| Method | Endpoint | Description | Sample Response |
| --- | --- | --- | --- |
| `GET`, `POST` | `/` | Main search interface for text and image queries. | HTML page |
| `GET` | `/search` | Returns FAISS cosine-similarity results as JSON. | `{ "results": [...] }` |
| `GET` | `/categories` | Returns available category labels. | `{ "categories": ["cat", "dog"] }` |
| `GET` | `/history` | Returns recent session query history. | `[ { "query": "dog", "timestamp": "..." } ]` |
| `POST` | `/search/multimodal` | Search with both `query_text` and `query_image`. | `{ "image_paths": [...], "results": [...] }` |
| `POST` | `/search/rerank` | Apply relevance feedback using relevant and irrelevant paths. | `{ "results": [...], "top_k": 5 }` |
| `POST` | `/favorites/add` | Save an image path to favorites. | `{ "success": true, "favorites": [...] }` |
| `POST` | `/favorites/remove` | Remove an image path from favorites. | `{ "success": true, "favorites": [...] }` |
| `POST` | `/admin/upload` | Upload new images and append them to the index. | `{ "added_count": 3, "failed_files": [] }` |
| `POST` | `/export/json` | Export selected results and similarity values as JSON. | Downloaded `results.json` file |
| `POST` | `/export/zip` | Export selected image files as a ZIP archive. | Downloaded `visionseek_results.zip` file |
| `GET` | `/image?path=...` | Resolve a stored path to an image response. | Image file or JSON error |
| `GET` | `/images/<path>` | Serve images directly from `dataset/images`. | Image file |
| `GET` | `/dataset/<path>` | Serve files from the dataset root. | File response |
| `GET` | `/placeholder.png` | Return a placeholder image when a result is missing. | SVG image |

Sample request for the text API:

```bash
curl -X POST "http://127.0.0.1:5000/api/v1/search/text" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query":"a white dog","top_k":5}'
```

## Screenshots

Add screenshots here once they are available:

- Home page / search interface: `docs/screenshots/home.png` - To Be Added
- Search results grid: `docs/screenshots/results.png` - To Be Added
- Multimodal / rerank flow: `docs/screenshots/rerank.png` - To Be Added
- Export and favorites workflow: `docs/screenshots/export-favorites.png` - To Be Added

## Model Architecture

VisionSeek is an inference-first semantic retrieval system. It does not train a custom model in this repository; instead, it uses a pretrained vision-language model and a vector index.

### Core model

- Encoder family: OpenCLIP
- Model name: `ViT-B-32`
- Pretrained tag: `openai`
- Embedding dimensionality: 512
- Normalization: L2 normalization before cosine search

### Search pipeline

1. User submits text, an image, or both.
2. The app encodes the input with OpenCLIP.
3. The embedding is normalized for cosine similarity.
4. FAISS `IndexFlatIP` performs inner-product lookup over normalized vectors.
5. Results are formatted with similarity scores and relative paths.
6. Optional category filtering, favorites decoration, and reranking are applied.

### Dataset information

- The codebase is built around a flattened image dataset stored in `dataset/images/`.
- Metadata is derived from filenames and persisted in `metadata.json`.
- The repository's report generator references an animal-image dataset with about 10 categories such as `dog`, `cat`, `bird`, `fish`, `elephant`, `horse`, `cow`, `chicken`, `spider`, and `squirrel`.
- To Be Added: exact dataset size, source, and provenance if you want a fully auditable training/data section.

### Training and evaluation

- Training: None in this repository. The app uses pretrained embeddings.
- Evaluation: interactive relevance feedback, similarity scores, and export flows.
- Metrics exposed in the app: cosine similarity converted to a 0-100 score.
- To Be Added: formal benchmark metrics such as top-k accuracy, precision@k, recall@k, or latency benchmarks.

## Future Improvements

- Externalize the admin upload token into environment variables.
- Add a `requirements.txt` or lock file for reproducible installs.
- Add automated tests for API routes and index-update behavior.
- Add Docker support for one-command local deployment.
- Persist favorites and history in a database instead of flat files if the project grows.
- Add formal evaluation metrics and a benchmark report.
- Add CI checks for linting, tests, and packaging.
- Add a screenshot set and optionally a short demo GIF.

## Contributing

1. Fork the repository and create a feature branch.
2. Make focused changes that preserve the current API and UI behavior.
3. Run the application locally and verify search, upload, export, and rerank flows.
4. Update documentation if your change affects setup, configuration, or endpoints.
5. Open a pull request with a concise description of the change and verification steps.

Suggested local checks before submitting a PR:

```bash
python app.py
python src/generate_embeddings.py
python scripts/test_search_similarity.py
```

## License

To Be Added.

There is currently no `LICENSE` file in the repository. If you want this project to be publicly reusable, add a license such as MIT or Apache 2.0 and commit the corresponding file.
