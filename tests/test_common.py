"""Standard-library-only tests for deterministic helper functions."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import image_path, metadata, positive_int, save_json

class TestCommon(unittest.TestCase):
    def test_positive_int(self):
        self.assertEqual(positive_int("3"), 3)
        with self.assertRaises(ValueError):
            positive_int("0")

    def test_image_missing(self):
        with self.assertRaises(FileNotFoundError):
            image_path("/nonexistent-vlm-test-input.jpg")

    def test_json_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "output.json"
            save_json(path, {"caption": "a café"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"caption": "a café"})

    def test_metadata_timestamp(self):
        self.assertIn("timestamp_utc", metadata())

if __name__ == "__main__":
    unittest.main()
