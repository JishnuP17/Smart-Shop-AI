"""Command-line rebuild for the persistent product FAISS index."""

from app.services.semantic_search import index_existing_products


if __name__ == "__main__":
    print(index_existing_products())
