"""Opt-in, bounded three-request Wan session against a dedicated local server.

Without --run, prints the planned graphs and makes no HTTP requests. The parent
must own the server process under gpu_runner.py --monitor; this client cannot
bound another process's GPU memory. No /free calls occur. On failure, only this
client's prompt is cancelled through the server's atomic job-cancel endpoint.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "templates_api" / "video_wan2_2_5B_ti2v.json"


def graphs(session):
    graph = json.loads(TEMPLATE.read_text())
    expected = {"3": "KSampler", "6": "CLIPTextEncode", "37": "UNETLoader",
                "38": "CLIPLoader", "39": "VAELoader", "55": "Wan22ImageToVideoLatent", "58": "SaveVideo"}
    for node, kind in expected.items():
        if graph[node]["class_type"] != kind:
            raise ValueError(f"Template contract changed at {node}")
    graph["37"]["inputs"]["unet_name"] = "wan2.2_ti2v_5b_turbo_int8_convrot.safetensors"
    graph["38"]["inputs"]["clip_name"] = "umt5_xxl_fp16.safetensors"
    graph["3"]["inputs"].update(seed=7, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple")
    graph["55"]["inputs"].update(width=704, height=416, length=33, batch_size=1)
    base_prompt = graph["6"]["inputs"]["text"]
    result = []
    for label, seed, prompt in (
        ("initial", 7, base_prompt),
        ("changed_seed", 8, base_prompt),
        ("changed_prompt", 8, base_prompt + " The musician wears a bright blue scarf."),
    ):
        current = copy.deepcopy(graph)
        current["3"]["inputs"]["seed"] = seed
        current["6"]["inputs"]["text"] = prompt
        current["58"]["inputs"]["filename_prefix"] = f"review/session-{session}/{label}"
        result.append((label, current))
    return result


def summarize(history, graph, url, output_dir):
    status = history.get("status") or {}
    messages = status.get("messages", [])
    by_type = {kind: body for kind, body in messages}
    has_cache_event = "execution_cached" in by_type
    cached = sorted({str(node) for kind, body in messages if kind == "execution_cached" for node in body.get("nodes", [])})
    files = []
    for node, output in history.get("outputs", {}).items():
        for kind in ("images", "gifs", "videos", "audio"):
            for item in output.get(kind, []):
                if not isinstance(item, dict) or "filename" not in item:
                    continue
                parameters = {key: item.get(key, default) for key, default in (
                    ("filename", ""), ("subfolder", ""), ("type", "output"))}
                record = {"node": node, **parameters, "view_url": url + "/view?" + urllib.parse.urlencode(parameters)}
                if output_dir and parameters["type"] == "output":
                    record["local_path"] = str(output_dir / parameters["subfolder"] / parameters["filename"])
                files.append(record)
    end = by_type.get("execution_success", by_type.get("execution_error", by_type.get("execution_interrupted", {})))
    begin = by_type.get("execution_start", {})
    span = ((end["timestamp"] - begin["timestamp"]) / 1000
            if "timestamp" in end and "timestamp" in begin else None)
    successful = status.get("status_str") == "success" and status.get("completed") is True
    ran_sampler = successful and has_cache_event and "3" not in cached
    return {"ok": ran_sampler, "status": status,
            "server_span_s": span, "cache_event_present": has_cache_event,
            "cached_nodes": cached if has_cache_event else None,
            "cached_classes": {node: graph[node]["class_type"] for node in cached if node in graph} if has_cache_event else None,
            "sampler_cache_hit": "3" in cached if has_cache_event else None,
            "sampler_execution_evidence": "successful graph with sampler absent from execution_cached" if ran_sampler else None,
            "positive_conditioning_cache_hit": "6" in cached if has_cache_event else None,
            "loader_cache_hits": {node: node in cached if has_cache_event else None for node in ("37", "38", "39")},
            "outputs": files,
            "error": by_type.get("execution_error", by_type.get("execution_interrupted"))}


class Client:
    def __init__(self, url, seconds, cancelled):
        self.url = url.rstrip("/")
        self.deadline = time.monotonic() + seconds
        self.cancelled = cancelled
        self.pending = None

    def request(self, path, data=None, cleanup=False):
        remaining = self.deadline - time.monotonic()
        if not cleanup and (remaining <= 0 or self.cancelled()):
            raise TimeoutError("Session deadline or cancellation")
        request = urllib.request.Request(self.url + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=3 if cleanup else min(5, remaining)) as response:
                return json.loads(response.read() or b"null")
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code} at {path}: {error.read(8192).decode(errors='replace')}") from error

    def run(self, graph, client_id):
        queue = self.request("/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("Dedicated server is not idle; refusing to add work")
        start = time.monotonic()
        prompt_id = str(uuid.uuid4())
        # Caller-selected ID is supported by this server, so cancellation still
        # has an identity when POST succeeds but its response is lost.
        self.pending = prompt_id
        submitted = self.request("/prompt", {"prompt": graph, "client_id": client_id, "prompt_id": prompt_id})
        if submitted.get("prompt_id") != prompt_id:
            raise RuntimeError("Server did not preserve requested prompt identity")
        while True:
            history = self.request("/history/" + urllib.parse.quote(prompt_id, safe=""))
            if prompt_id in history:
                self.pending = None
                return prompt_id, time.monotonic() - start, history[prompt_id]
            time.sleep(min(0.5, max(0, self.deadline - time.monotonic())))

    def cleanup(self):
        if self.pending is None:
            return {"needed": False}
        prompt_id = self.pending
        result = self.request("/api/jobs/" + urllib.parse.quote(prompt_id, safe="") + "/cancel", {}, cleanup=True)
        # Cancellation acknowledgment is not proof that GPU work has stopped.
        until = time.monotonic() + 10
        while time.monotonic() < until:
            queue = self.request("/queue", cleanup=True)
            present = any(str(item[1]) == prompt_id for key in ("queue_running", "queue_pending") for item in queue.get(key, []))
            if not present:
                self.pending = None
                return {"needed": True, "prompt_id": prompt_id, "cancel_response": result, "left_queue": True}
            time.sleep(0.2)
        return {"needed": True, "prompt_id": prompt_id, "cancel_response": result, "left_queue": False,
                "action": "Parent must stop its owned server process group"}


def log_delta(path, offset):
    if path is None:
        return {"available": False}, offset
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read(4 * 1024**2)
        end = stream.tell()
    text = raw.decode(errors="replace")
    phrases = {"requested_load": "Requested to load ", "loaded_complete": "loaded completely;",
               "loaded_partial": "loaded partially;", "unloaded_partial": "Unloaded partially:", "unloading": "Unloading "}
    return {"available": True, "counts": {name: text.count(phrase) for name, phrase in phrases.items()},
            "transitions": [line for line in text.splitlines() if any(phrase in line for phrase in phrases.values())],
            "truncated": len(raw) == 4 * 1024**2}, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8188"))
    parser.add_argument("--seconds", type=float, default=480)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--report", type=Path, default=HERE.parent / "logs" / "review-server-session.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("MK_ROOT", os.path.expanduser("~/mac-kernels-data"))) / "out" / "comfyui")
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.query or parsed.fragment:
        parser.error("Use a dedicated local HTTP ComfyUI server")
    if not 0 < args.seconds <= 520:
        parser.error("Session deadline must be positive and at most 520 seconds, leaving cleanup time")
    session = uuid.uuid4().hex[:12]
    planned = graphs(session)
    if not args.run:
        print(json.dumps({"run": False, "session": session, "graphs": planned}, indent=2))
        return 0
    cancelled = [False]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda number, frame: cancelled.__setitem__(0, True))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def emit(event, **data):
        line = json.dumps({"event": event, "session": session, **data})
        print(line, flush=True)
        with args.report.open("a") as stream:
            stream.write(line + "\n")
    client = Client(args.url, args.seconds, lambda: cancelled[0])
    offset = args.server_log.stat().st_size if args.server_log else 0
    exit_code = 0
    try:
        emit("session_start", url=args.url, seconds=args.seconds,
             settings="Wan5B Turbo INT8,704x416x33,4steps,Euler/simple,CFG1,fullVAE; no /free between requests",
             initial_server=client.request("/system_stats"), cold_claim="Initial is cold only if server was freshly started; cache evidence follows")
        for label, graph in planned:
            before = client.request("/system_stats")
            prompt_id, wall, history = client.run(graph, session)
            result = summarize(history, graph, args.url, args.output_dir)
            transitions, offset = log_delta(args.server_log, offset)
            emit("request", label=label, prompt_id=prompt_id, wall_s=wall,
                 seed=graph["3"]["inputs"]["seed"], prompt=graph["6"]["inputs"]["text"],
                 graph_sha256=hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest(),
                 before=before, after=client.request("/system_stats"), log=transitions, **result)
            if not result["ok"]:
                raise RuntimeError("Graph failed, sampler was cached, or cache evidence is missing; sequence is not a valid generation comparison")
    except (OSError, ValueError, RuntimeError) as error:
        exit_code = 1
        emit("error", message=str(error))
    finally:
        try:
            cleanup = client.cleanup()
            if cleanup.get("left_queue") is False:
                exit_code = 1
            emit("cleanup", **cleanup, server_left_running=True,
                 ownership="Parent owns final server teardown; this client never frees unrelated models or cancels other jobs")
        except (OSError, ValueError, RuntimeError) as error:
            exit_code = 1
            emit("cleanup_error", message=str(error), action="Parent must stop its owned server process group")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
