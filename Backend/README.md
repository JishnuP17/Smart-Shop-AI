# Personalized E-Commerce Search API (Initial Backend)

This folder contains a FastAPI product catalog API with local semantic search. The pretrained `sentence-transformers/all-MiniLM-L6-v2` model creates normalized 384-dimensional embeddings. FAISS performs cosine ranking; MongoDB stores product records and supplies the matching records by stable ID.

## Requirements

- Python 3.10 or newer
- A MongoDB instance and a database containing a `products` collection
- Visual Studio Code (recommended)

## Set up on Windows with VS Code and PowerShell

Open this `Backend` folder in VS Code. In VS Code, open **Terminal → New Terminal** and confirm the terminal is PowerShell. Run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set the MongoDB values for your environment:

```dotenv
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=ecommerce
```

For a hosted MongoDB service, put its connection URI in `MONGODB_URI`. Keep `.env` private; it is excluded by `.gitignore`. Never commit credentials.

If PowerShell blocks virtual environment activation, allow it for the current terminal session, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the API

From this `Backend` folder, with the virtual environment active:

```powershell
uvicorn app.main:app --reload
```

The server listens at `http://127.0.0.1:8000`. Stop it with **Ctrl+C**.

## Endpoints

- `GET /` — welcome message
- `GET /api/health` — API process status
- `GET /api/health/db` — MongoDB ping status; returns HTTP 503 if MongoDB is unreachable or not configured
- `GET /api/products?skip=0&limit=20&category=Electronics` — read a page of catalog products, optionally filtered by exact category
- `GET /api/products/categories` — list categories and counts derived from stored products
- `POST /api/products` — create a product and generate its embedding
- `GET /api/products/search?q=comfortable%20trail%20shoes&limit=10` — rank products by semantic similarity
- `GET /api/products/{product_id}/recommendations?limit=5` — recommend similar products by selected product ID
- `POST /api/products/index-embeddings` — rebuild the persistent FAISS index from the catalog

Open `http://127.0.0.1:8000/docs` to explore and try the endpoints interactively. Product pagination accepts `skip` values of 0 or greater and `limit` values from 1 to 100. Product responses omit internal embedding fields.

## Configuration and database behavior

`MONGODB_URI`, `MONGODB_DATABASE`, and optional `EMBEDDING_MODEL` are loaded from environment variables or `.env`. The shared PyMongo client is reused. The FAISS index and its MongoDB ID mapping are saved under `Backend/data/faiss/` in immutable generations; `.gitignore` excludes those generated files. The index loads lazily on first search and is not rebuilt per query. API responses hide internal vector metadata and serialize ObjectIds as strings.

The first embedding request downloads the local model from Hugging Face and caches it; internet access is needed for that initial download. Search uses FAISS `IndexFlatIP` with normalized vectors (cosine similarity), while MongoDB is only used for product data. Search checks a signature of the product IDs and semantic text against the saved index and returns a clear rebuild error if they differ. After adding or changing products, rebuild with `python -m app.services.vector_search --rebuild` or `POST /api/products/index-embeddings`.

## Content-based recommendations

Content-based recommendations find catalog items whose existing product embeddings are closest to a selected product's vector. The endpoint reuses the persistent FAISS index and its ID mapping, fetches the selected and recommended products from MongoDB, excludes the selected item, removes duplicate IDs, and returns results in cosine-similarity order. Scores are cosine similarity values (not percentages). It does not use user history or collaborative filtering.

Example request:

```text
GET http://127.0.0.1:8000/api/products/PRODUCT_ID/recommendations?limit=5
```

Replace `PRODUCT_ID` with a MongoDB `_id` returned by `GET /api/products`. The response includes `product_id`, `indexed_products`, and `recommendations`; each recommendation contains `product` fields such as `_id`, `name`, `price`, `category`, `brand`, `description`, and `image_url` when present, plus a cosine `score`. Internal embedding values are never returned. A missing product or product without a vector returns 404; a missing, stale, or inconsistent index returns 503. Use Swagger at `http://127.0.0.1:8000/docs`, open **GET /api/products/{product_id}/recommendations**, enter an ID from the products endpoint, then execute. The index has a vector for each currently indexed catalog record; new products need to be indexed before recommendation requests can use them.

This is a content-only baseline: it does not account for popularity, availability, user preferences, or collaborative signals. Similarity reflects the fields used when the product vector was generated.

## Test the search

Run `python -m unittest discover -s tests -v` from this folder. Build or rebuild the index with `python -m app.services.vector_search --rebuild`. In Postman, search with `GET http://127.0.0.1:8000/api/products/search?q=comfortable%20trail%20shoes&limit=10`. The returned score is a cosine similarity value, not a probability.

## Import the Amazon Products 2023 dataset

The importer reads the public `iguzelofficial/AMAZON-Products-2023` train split from Hugging Face's Parquet conversion. The dataset card reports 117,243 rows and 15 source columns. The importer projects only `parent_asin`, `title`, `description`, `main_category`, `categories`, `store`, `price`, and `image`; it ignores the dataset's supplied embeddings because they were generated by a different model. It maps `title` to `name`, `main_category` (or `categories`) to `category`, `store` to `brand`, and `image` to `image_url`. The source has no separate brand field, so `store` is the available approximation.

The script fetches Parquet shard metadata and uses HTTP byte ranges to read selected columns. It does not save the full dataset locally; a small bounded byte-block cache lives in memory. The pretrained MiniLM model remains in Hugging Face's normal user cache. No import runs automatically.

From this `Backend` folder in PowerShell, first run a read-only test that downloads and embeds 100 rows without connecting to or writing MongoDB:

```powershell
.\.venv\Scripts\python.exe -m app.services.import_products --limit 100 --batch-size 32 --dry-run
```

After reviewing a successful dry run, import the same 100-row subset into the configured database:

```powershell
.\.venv\Scripts\python.exe -m app.services.import_products --limit 100 --batch-size 32
```

The summary reports `scanned`, `imported`, `updated`, `skipped`, `rejected`, `failed`, and `invalid_price`. Products with missing or invalid prices are retained without a `price`; malformed records without a stable `parent_asin` or usable title are rejected. Repeated IDs and unchanged products from prior runs are skipped. Existing unrelated product documents are preserved. A partial unique index scopes `source_dataset` plus `source_product_id` to this dataset; before creating it, the importer checks for missing or duplicate IDs in already dataset-scoped records and stops safely if any are found. Only the importer creates this index when you run it.

Importing generates embeddings in batches and temporarily stores them on the imported MongoDB documents. After the FAISS rebuild succeeds, vector arrays are saved in the FAISS index and removed from MongoDB; the content hash remains for safe idempotent imports. Search will report a stale index after product data changes until the rebuild command is run.

To import up to 5,000 source rows later, use:

```powershell
.\.venv\Scripts\python.exe -m app.services.import_products --limit 5000 --batch-size 32
```

Increase `--batch-size` to process more texts per model call, or lower it to reduce peak memory use. Rerunning the same command is safe: unchanged products with valid MiniLM embeddings are skipped.

### Verify an import

In MongoDB Compass, open `ecommerce_ai.products` and filter for:

```json
{"source_dataset": "iguzelofficial/AMAZON-Products-2023"}
```

After a successful FAISS rebuild, embeddings should be in the local index rather than product documents. The persistent mapping is in `Backend/data/faiss/generations/<generation>/mapping.json`; each mapping entry records the vector position, MongoDB ID and type, plus dataset scope and source ID. To check the temporary embedding fields before building the index, use:

```json
{
  "source_dataset": "iguzelofficial/AMAZON-Products-2023",
  "_embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

Before the index is built, each `_embedding` array should contain 384 values. After a successful build, verify that a generation exists under `Backend/data/faiss/generations/` and its manifest reports the expected vector count and dimension. API responses hide `_embedding`, `_embedding_model`, and `_source_content_hash`. In Postman, verify `GET http://127.0.0.1:8000/api/products?limit=5`, then try natural-language searches such as:

```text
GET http://127.0.0.1:8000/api/products/search?q=lightweight%20running%20shoes&limit=10
GET http://127.0.0.1:8000/api/products/search?q=wireless%20audio%20for%20travel&limit=10
GET http://127.0.0.1:8000/api/products/search?q=non-toxic%20craft%20supplies&limit=10
```

The dataset card declares the dataset license as MIT and attributes the underlying product metadata to Amazon Reviews 2023. The dataset page also describes its intended research/search uses and provides a paper citation. Review the dataset card and the source project's terms before redistributing imported Amazon catalog metadata or using it commercially; the card's license label does not replace review of the upstream data source's terms.
