import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from eval_r3_processed_answers_api_v6 import (
    SingleFlightExtractionCache,
    extraction_cache_key,
    extraction_from_scored_record,
)


class SingleFlightExtractionCacheTest(unittest.TestCase):
    def test_duplicate_concurrent_requests_call_backend_once(self):
        lock = threading.Lock()
        calls = 0

        def backend(client, **kwargs):
            del client, kwargs
            nonlocal calls
            with lock:
                calls += 1
            time.sleep(0.02)
            return {
                "ok": True,
                "candidates": ["answer"],
                "raw_output": '["answer"]',
                "attempts": 1,
                "error_type": "",
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }

        cache = SingleFlightExtractionCache(backend)
        kwargs = {
            "model": "test-model",
            "question": "Same question?",
            "answer": "Same answer.",
            "max_tokens": 32,
            "retries": 0,
            "provider_routing": False,
        }
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: cache(None, **kwargs), range(4)))

        self.assertEqual(calls, 1)
        self.assertEqual(sum(not result["cache_hit"] for result in results), 1)
        self.assertEqual(sum(result["usage"]["total_tokens"] for result in results), 12)

    def test_key_is_question_and_prediction_specific(self):
        self.assertEqual(extraction_cache_key("q", "a"), extraction_cache_key("q", "a"))
        self.assertNotEqual(extraction_cache_key("q", "a"), extraction_cache_key("q", "b"))

    def test_completed_scored_record_can_seed_resume(self):
        extraction = extraction_from_scored_record(
            {
                "status": "extracted",
                "candidate_answers": ["A"],
                "extractor_output": '["A"]',
                "extractor_attempts": 1,
                "extractor_error_type": "",
                "usage": {"total_tokens": 5},
            }
        )
        self.assertIsNotNone(extraction)
        self.assertEqual(extraction["candidates"], ["A"])


if __name__ == "__main__":
    unittest.main()
