"""Run the TLS RAG CLI from a source checkout, including without installation."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from tls_rag.cli import main

    main()
