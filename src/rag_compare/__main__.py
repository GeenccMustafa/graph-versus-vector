"""Allow ``python -m rag_compare ...`` as an alternative to the console script."""

from .cli import app

if __name__ == "__main__":
    app()
