#!/usr/bin/env python3
"""Entry point for the LM_LaLiga web dashboard."""
import uvicorn
from pathlib import Path
import sys

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
