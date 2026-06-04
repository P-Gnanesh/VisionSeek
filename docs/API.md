# VisionSeek API

Base path: `/api/v1`

## Authentication

If the `VISIONSEEK_API_KEY` environment variable is set, all API routes require an `X-API-Key` header whose value matches that key.

Example:

```bash
X-API-Key: your-api-key
```

## `POST /api/v1/search/text`

Search by natural-language text.

### Request schema

```json
{
  "query": "string, required",
  "top_k": "integer, optional, default 5"
}
```

### Example curl

```bash
curl -X POST "http://localhost:5000/api/v1/search/text" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"query":"a red car","top_k":5}'
```

### Response schema

```json
{
  "query": "string",
  "results": [
    {
      "image_path": "string",
      "filename": "string",
      "score": "number"
    }
  ],
  "elapsed_ms": "number"
}
```

### Error codes

`400` invalid JSON body, missing `query`, or invalid `top_k`.

`401` missing or invalid `X-API-Key` when `VISIONSEEK_API_KEY` is configured.

## `POST /api/v1/search/image`

Search by uploaded image.

### Request schema

Multipart form-data with:

```text
image: file, required
top_k: integer, optional, default 5
```

### Example curl

```bash
curl -X POST "http://localhost:5000/api/v1/search/image" \
  -H "X-API-Key: your-api-key" \
  -F "image=@C:/path/to/query.jpg" \
  -F "top_k=5"
```

### Response schema

```json
{
  "query": "string",
  "results": [
    {
      "image_path": "string",
      "filename": "string",
      "score": "number"
    }
  ],
  "elapsed_ms": "number"
}
```

### Error codes

`400` missing `image` upload or invalid `top_k`.

`401` missing or invalid `X-API-Key` when `VISIONSEEK_API_KEY` is configured.

## `GET /api/v1/health`

Returns the current service status and the number of indexed images.

### Example curl

```bash
curl "http://localhost:5000/api/v1/health" \
  -H "X-API-Key: your-api-key"
```

### Response schema

```json
{
  "status": "ok",
  "index_size": "integer"
}
```

### Error codes

`401` missing or invalid `X-API-Key` when `VISIONSEEK_API_KEY` is configured.