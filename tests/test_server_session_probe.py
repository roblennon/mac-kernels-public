"""Client-contract checks using a local HTTP fixture; no ComfyUI or GPU starts."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from probe_server_session import Client, graphs, summarize


class SessionHTTPTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.submitted = []
        self.pending = None
        self.never_finishes = False
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def answer(self, value):
                encoded = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):
                owner.requests.append(self.path)
                if self.path == "/queue":
                    running = [[0, owner.pending]] if owner.pending else []
                    self.answer({"queue_running": running, "queue_pending": []})
                elif self.path.startswith("/history/"):
                    if owner.never_finishes:
                        self.answer({})
                        return
                    pid = self.path.rsplit("/", 1)[1]
                    owner.pending = None
                    index = len(owner.submitted)
                    cached = [] if index == 1 else ["37", "38", "39"] + (["6"] if index == 2 else [])
                    self.answer({pid: {"status": {"completed": True, "status_str": "success",
                        "messages": [["execution_start", {"timestamp": 1000}],
                                     ["execution_cached", {"nodes": cached}],
                                     ["execution_success", {"timestamp": 2000}]]},
                        "outputs": {"58": {"videos": [{"filename": "clip.mp4", "subfolder": "review", "type": "output"}]}}}})
                else:
                    self.send_error(404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.requests.append(self.path)
                if self.path == "/prompt":
                    owner.submitted.append(body)
                    owner.pending = body["prompt_id"]
                    self.answer({"prompt_id": owner.pending})
                elif self.path.endswith("/cancel"):
                    expected = f"/api/jobs/{owner.pending}/cancel"
                    if self.path != expected:
                        self.send_error(409)
                        return
                    owner.pending = None
                    self.answer({"cancelled": True})
                else:
                    self.send_error(404)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_real_http_sequence_changes_seed_then_prompt_without_free(self):
        client = Client(self.url, 5, lambda: False)
        results = []
        for label, graph in graphs("cpu-test"):
            pid, wall, history = client.run(graph, "cpu-test")
            result = summarize(history, graph, self.url, Path("/outputs"))
            self.assertTrue(result["ok"])
            self.assertEqual(result["server_span_s"], 1)
            self.assertEqual(result["outputs"][0]["local_path"], "/outputs/review/clip.mp4")
            results.append(result)
        self.assertEqual([body["prompt"]["3"]["inputs"]["seed"] for body in self.submitted], [7, 8, 8])
        prompts = [body["prompt"]["6"]["inputs"]["text"] for body in self.submitted]
        self.assertEqual(prompts[0], prompts[1])
        self.assertNotEqual(prompts[1], prompts[2])
        self.assertEqual([result["positive_conditioning_cache_hit"] for result in results], [False, True, False])
        self.assertNotIn("/free", self.requests)
        self.assertEqual(client.cleanup(), {"needed": False})

    def test_deadline_cancels_only_the_owned_prompt_and_verifies_departure(self):
        self.never_finishes = True
        client = Client(self.url, 5, lambda: False)
        request = client.request
        def expire_after_submission(path, *args, **kwargs):
            result = request(path, *args, **kwargs)
            if path == "/prompt":
                client.deadline = 0
            return result
        client.request = expire_after_submission
        with self.assertRaises(TimeoutError):
            client.run(graphs("timeout")[0][1], "timeout")
        owned = client.pending
        self.assertIsNotNone(owned)
        self.assertTrue(client.cleanup()["left_queue"])
        self.assertIn(f"/api/jobs/{owned}/cancel", self.requests)
        self.assertNotIn("/interrupt", self.requests)

    def test_cached_sampler_cannot_be_reported_as_generated_video(self):
        graph = graphs("cache")[0][1]
        history = {"status": {"status_str": "success", "completed": True,
                   "messages": [["execution_cached", {"nodes": ["3"]}]]}}
        result = summarize(history, graph, self.url, None)
        self.assertFalse(result["ok"])
        self.assertTrue(result["sampler_cache_hit"])

    def test_missing_cache_event_leaves_sampler_execution_unknown(self):
        graph = graphs("unknown")[0][1]
        history = {"status": {"status_str": "success", "completed": True,
                   "messages": [["execution_success", {"timestamp": 2000}]]}}
        result = summarize(history, graph, self.url, None)
        self.assertFalse(result["ok"])
        self.assertFalse(result["cache_event_present"])
        self.assertIsNone(result["sampler_cache_hit"])
        self.assertIsNone(result["sampler_execution_evidence"])
        self.assertIsNone(result["positive_conditioning_cache_hit"])
        self.assertEqual(result["loader_cache_hits"], {"37": None, "38": None, "39": None})


if __name__ == "__main__":
    unittest.main()
