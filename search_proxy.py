"""
SearXNG-compatible search proxy — no Docker needed.
Uses DuckDuckGo lite (more stable HTML than full DDG).
"""
import json, re, time, urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 8080

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def ddg_search(query: str, max_results: int = 8) -> list:
    # Use DDG lite — simpler HTML, more stable
    url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote_plus(query)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            html = r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[proxy] fetch error: {e}")
        return []

    results = []
    # DDG lite: results in <tr> rows with <a> links
    rows = re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>.*?<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</td>',
                      html, re.DOTALL)
    for href, title, snippet in rows[:max_results]:
        title   = re.sub(r'<[^>]+>', '', title).strip()
        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        if title and href:
            results.append({"title": title, "url": href,
                            "content": snippet, "engine": "duckduckgo"})

    # Fallback: grab any links if above fails
    if not results:
        for href, title in re.findall(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', html):
            title = re.sub(r'<[^>]+>', '', title).strip()
            if title and len(title) > 10 and 'duckduckgo' not in href:
                results.append({"title": title, "url": href, "content": "", "engine": "duckduckgo"})
                if len(results) >= max_results: break

    return results

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass  # silence access log

    def do_GET(self):
        if self.path == "/healthz":
            self._json(200, {"status": "ok", "engine": "duckduckgo-proxy"})
        elif self.path.startswith("/search"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            q  = qs.get("q", [""])[0]
            t0 = time.time()
            results = ddg_search(q) if q else []
            ms = int((time.time()-t0)*1000)
            print(f"[proxy] '{q}' → {len(results)} results in {ms}ms")
            self._json(200, {"query": q, "results": results,
                             "number_of_results": len(results)})
        else:
            self._json(404, {"error": "not found"})

    def _json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[proxy] SearXNG-compatible proxy on http://127.0.0.1:{PORT}")
    print(f"[proxy] Set env: export SEARXNG_URL=http://localhost:{PORT}")
    server.serve_forever()
