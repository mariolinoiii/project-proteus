"""Project Proteus - ProteusSegNet standalone local application (loopback only, single user)."""
import os, sys, json, mimetypes, threading, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import services as S

WEB = (Path(__file__).resolve().parent / "web").resolve()
PORT = int(os.environ.get("PROTEUS_PORT", "8765"))
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "ProteusSegNet/3"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, ctype=None, download=None):
        path = Path(path)
        if not path.is_file():
            return self._send(404, {"error": "Not found"})
        self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.end_headers()
        with open(path, "rb") as h:
            while chunk := h.read(1 << 20):
                self.wfile.write(chunk)

    def _guard(self):
        if self.headers.get("Host") not in ALLOWED_HOSTS:
            self._send(403, {"error": "Forbidden host"}); return False
        origin = self.headers.get("Origin")
        if self.command == "POST" and origin and urlparse(origin).netloc not in ALLOWED_HOSTS:
            self._send(403, {"error": "Forbidden origin"}); return False
        return True

    def do_GET(self):
        if not self._guard():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        path = unquote(u.path)
        try:
            if path in ("/", "/index.html"):
                return self._file(WEB / "index.html")
            if path.startswith("/static/"):
                f = (WEB / path[8:]).resolve()
                return self._file(f) if WEB in f.parents else self._send(404, {"error": "Not found"})
            if path == "/api/state":
                return self._send(200, S.state())
            if path == "/api/diagnostics":
                return self._send(200, S.diagnostics())
            if path.startswith("/api/job/"):
                job = S.get_job(path.rsplit("/", 1)[1])
                return self._send(200, job) if job else self._send(404, {"error": "Unknown job"})
            if path == "/api/inspect":
                return self._send(200, S.inspect_sar(q["name"]))
            if path == "/api/dataset/scenes":
                ds = S.load_latest_dataset()
                return self._send(200, {"dataset": ds, "scenes": S.dataset_scenes(ds["index_path"]) if ds else []})
            if path == "/api/dataset/preview":
                ds = S.load_latest_dataset()
                row = next(r for r in S.dataset_scenes(ds["index_path"]) if r["image_id"] == q["scene"])
                import previews
                out = Path(ds["index_path"]).parent / "previews"
                sid = row["image_id"]
                if (out / f"{sid}_label_chart.png").is_file() and (out / f"{sid}_stats.json").is_file():
                    res = {"image_id": sid, "stats": json.loads((out / f"{sid}_stats.json").read_text())}
                else:
                    res = previews.render_scene(row, S.CACHE, out)
                rel = out.relative_to(S.OUT).as_posix()
                res["urls"] = {k: f"/files/{rel}/{sid}_{k}.png" for k in ("sar", "worldcover", "labels", "overlay", "worldcover_chart", "label_chart")}
                return self._send(200, res)
            if path == "/api/dataset/zip":
                f, ctype, name = S.dataset_zip()
                return self._file(f, ctype, name)
            if path.startswith("/api/runs/"):
                _, _, _, rid, kind = path.split("/")
                f, ctype, name = S.run_file(rid, kind)
                return self._file(f, ctype, name)
            if path == "/api/checkpoint":
                return self._file(S.MODEL_PATH, "application/octet-stream", "proteus_model.pt")
            if path.startswith("/files/"):
                f = (S.OUT / path[7:]).resolve()
                if S.OUT.resolve() in f.parents and f.suffix.lower() == ".png":
                    return self._file(f)
            return self._send(404, {"error": "Not found"})
        except StopIteration:
            return self._send(404, {"error": "Scene not found"})
        except Exception as exc:
            S.log_error(f"GET {self.path}: {exc}")
            return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        if not self._guard():
            return
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/upload":
                return self._send(200, {"name": S.save_upload(q["kind"], q["name"], self.rfile, int(self.headers.get("Content-Length", 0)))})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            routes = {"/api/analysis/run": ("analysis", S.run_analysis_job), "/api/dataset/prepare": ("dataset", S.prepare_dataset_job),
                      "/api/train/start": ("training", S.train_job)}
            if u.path in routes:
                kind, fn = routes[u.path]
                return self._send(200, {"job": S.start_job(kind, fn, p=body)})
            if u.path == "/api/shutdown":
                if S.active_job():
                    return self._send(409, {"error": "A job is active; it will not be stopped."})
                self._send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            return self._send(404, {"error": "Not found"})
        except Exception as exc:
            S.log_error(f"POST {self.path}: {exc}")
            return self._send(400, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Project Proteus v3 running at {url}  (workspace: {S.HOME})", flush=True)
    if os.environ.get("PROTEUS_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Proteus stopped.")


if __name__ == "__main__":
    main()
