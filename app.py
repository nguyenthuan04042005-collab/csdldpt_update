"""Local dashboard for the saved Chapter 3 features and KTH videos."""

import argparse
from base64 import b64encode
from hmac import compare_digest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Address, IPv4Network
import json
from pathlib import Path
import re
from secrets import token_urlsafe
import sqlite3
from urllib.parse import parse_qs, unquote, urlsplit

import cv2

from src.config import DATASET, DATABASE_PATH, WEB, feature_folder
from src.database.repository import RATINGS
from src.retrieval.similarity import METRICS
from src.services.query_service import QueryService
from src.visualization.visualizer import poster

RADMIN_NETWORK = IPv4Network("26.0.0.0/8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    static = {"/": ("index.html", "text/html; charset=utf-8"),
              "/app.css": ("app.css", "text/css; charset=utf-8"),
              "/app.js": ("app.js", "text/javascript; charset=utf-8")}

    @property
    def library(self):
        return self.server.query_service.engine

    def respond(self, code, content, mime, **headers):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(content)

    def send_json(self, data, code=200):
        self.respond(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                     "application/json; charset=utf-8", Cache_Control="no-store")

    def has_access(self):
        token = self.server.viewer_token
        if token is None:
            return True
        if IPv4Address(self.client_address[0]) not in RADMIN_NETWORK:
            self.respond(403, b"", "text/plain", Cache_Control="no-store", Connection="close")
            self.close_connection = True
            return False
        offered = self.headers.get("Authorization", "").encode("utf-8")
        if compare_digest(offered, token):
            return True
        self.respond(401, b"", "text/plain", Cache_Control="no-store", Connection="close",
                     WWW_Authenticate='Basic realm="Kho video", charset="UTF-8"')
        self.close_connection = True
        return False

    def video_path(self, key):
        path = (DATASET / key).resolve()
        if key not in self.library.index or not path.is_relative_to(DATASET) or not path.is_file():
            raise FileNotFoundError
        return path

    def send_video(self, path):
        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range")
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
            if match:
                low, high = match.groups()
                if low:
                    start = int(low)
                    end = min(end, int(high)) if high else end
                elif high:
                    start = max(0, size - int(high))
            if not match or not (match[1] or match[2]) or start > end or start >= size:
                self.respond(416, b"", "video/mp4", Content_Range=f"bytes */{size}")
                return
        self.send_response(206 if header else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command != "HEAD":
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    data = stream.read(min(remaining, 64 * 1024))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)

    def do_GET(self):
        if not self.has_access():
            return
        url = urlsplit(self.path)
        args = parse_qs(url.query)
        try:
            if url.path in self.static:
                name, mime = self.static[url.path]
                self.respond(200, (WEB / name).read_bytes(), mime, Cache_Control="no-cache")
            elif url.path == "/api/videos":
                self.send_json({"videos": self.library.items, "default": self.library.default,
                                "viewer": self.server.viewer_token is not None})
            elif url.path == "/api/search":
                key = args.get("query", [self.library.default])[0]
                metric = args.get("metric", ["manhattan"])[0]
                if key not in self.library.index or metric not in METRICS:
                    self.send_json({"error": "Video hoặc thước đo không hợp lệ."}, 400)
                else:
                    options = (key, metric, args.get("d", ["1"])[0] == "1", args.get("other", ["0"])[0] == "1")
                    self.send_json(self.server.query_service.search(
                        *options, read_only=self.server.viewer_token is not None
                    ))
            elif url.path == "/api/ratings.csv":
                if self.server.viewer_token is not None:
                    self.send_error(HTTPStatus.FORBIDDEN)
                else:
                    self.respond(200, self.server.query_service.repository.rating_csv(), "text/csv; charset=utf-8",
                                 Content_Disposition='attachment; filename="nhan_xet_video.csv"')
            elif url.path.startswith("/media/"):
                self.send_video(self.video_path(unquote(url.path[7:])))
            elif url.path.startswith("/poster/"):
                content = poster(self.video_path(unquote(url.path[8:])))
                self.respond(200, content, "image/jpeg", Cache_Control="private, max-age=3600")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    do_HEAD = do_GET

    def do_POST(self):
        if not self.has_access():
            return
        if self.server.viewer_token is not None:
            self.close_connection = True
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if urlsplit(self.path).path != "/api/rating":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 8192 or self.headers.get_content_type() != "application/json":
                raise ValueError
            body = json.loads(self.rfile.read(length))
            query, candidate, verdict = body["query"], body["candidate"], body["verdict"]
            if (not isinstance(query, str) or not isinstance(candidate, str) or query == candidate
                    or query not in self.library.index or candidate not in self.library.index
                    or (verdict is not None and verdict not in RATINGS)):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": "Nhận xét không hợp lệ."}, 400)
            return
        self.send_json({"rating_count": self.server.query_service.repository.rate(query, candidate, verdict)})

    def log_message(self, fmt, *args):
        if fmt.startswith("code"):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8061)
    parser.add_argument("--db", type=Path, default=DATABASE_PATH)
    parser.add_argument("--vpn-address", type=IPv4Address,
                        help="Bind to this machine's Radmin VPN address with a read-only account")
    args = parser.parse_args()
    if args.vpn_address is not None and args.vpn_address not in RADMIN_NETWORK:
        parser.error("--vpn-address must be the IPv4 address assigned by Radmin VPN")
    try:
        service = QueryService(args.db.resolve(), feature_folder("full"))
    except (sqlite3.Error, OSError, ValueError) as error:
        parser.error(f"Kho video chua san sang: {error}")
    cv2.setNumThreads(1)
    address = str(args.vpn_address) if args.vpn_address is not None else "127.0.0.1"
    with ThreadingHTTPServer((address, args.port), Handler) as server:
        server.query_service = service
        server.viewer_token = None
        if args.vpn_address is not None:
            password = token_urlsafe(18)
            server.viewer_token = b"Basic " + b64encode(f"viewer:{password}".encode("ascii"))
            print(f"Viewer: http://{address}:{server.server_port}/\nUsername: viewer\nPassword: {password}", flush=True)
        else:
            print(f"Local: http://{address}:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
