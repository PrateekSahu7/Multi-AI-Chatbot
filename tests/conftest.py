import os
import sys
import tempfile
from pathlib import Path

# Must be set before the app is imported.
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ["MOCK_DELAY_SCALE"] = "0"
os.environ["KEYS_FILE"] = str(Path(tempfile.mkdtemp()) / "keys.json")
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
