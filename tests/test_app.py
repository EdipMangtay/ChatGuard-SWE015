import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from chatguard.app import MASK, create_app
from chatguard.models import MODEL_NAMES, ModelBank
from chatguard.quantize import Int8Embedding, Int8Linear
from chatguard.text import distance, normalize, rule_label


class NormalizationTests(unittest.TestCase):
    def test_noisy_text(self):
        self.assertEqual(normalize("ÇOOOOK f0ol @Name https://example.com"), "cook fool usertoken urltoken")
        self.assertEqual(normalize("B@stard"), "bastard")

    def test_lexicon(self):
        self.assertEqual(distance("fukking", "fucking"), 1)
        self.assertEqual(rule_label("fukking"), 1)
        self.assertEqual(rule_label("good game everyone"), 0)


@unittest.skipUnless(Path("artifacts/manifest.json").exists(), "Run training or unpack weights.")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(Path("artifacts"))
        cls.client = TestClient(cls.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_default_and_quantization(self):
        health = self.client.get("/api/health").json()
        self.assertEqual(health["default_model"], "int8")
        bank: ModelBank = self.app.state.bank
        self.assertIsInstance(bank.int8.classifier, Int8Linear)
        self.assertIsInstance(bank.int8.bert.embeddings.word_embeddings, Int8Embedding)

    def test_mask_and_clean(self):
        clean = "Good game, everyone. Let's try the left route next round."
        bad = "You are a fukking idiot, uninstall the game."
        result = self.client.post("/api/moderate", json={"text": clean}).json()
        self.assertEqual((result["model"], result["label"], result["display_text"]), ("int8", "clean", clean))
        result = self.client.post("/api/moderate", json={"text": bad}).json()
        self.assertEqual((result["label"], result["display_text"]), ("offensive", MASK))
        self.assertGreaterEqual(result["latency_ms"], 0)

    def test_models_and_inputs(self):
        self.assertIn("Moderation Lab", self.client.get("/").text)
        self.assertIn("appendResult", self.client.get("/static/chat.js").text)
        for model in MODEL_NAMES:
            with self.subTest(model=model):
                result = self.client.post("/api/moderate", json={"text": "See you next match.", "model": model})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["model"], model)
                self.assertTrue(0.5 <= result.json()["confidence"] <= 1.0)
        for request in ({"text": ""}, {"text": " "}, {"text": "a" * 601},
                        {"text": "hello", "model": "invalid"}):
            self.assertEqual(self.client.post("/api/moderate", json=request).status_code, 422)


if __name__ == "__main__":
    unittest.main()
