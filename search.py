import urllib.request, urllib.parse, json, os, time, logging
log = logging.getLogger(__name__)
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")

def tool_search_multi(query, max_results=5, retries=2):
    if not isinstance(max_results, int) or max_results < 1:
        max_results = 5
    params = urllib.parse.urlencode({"q": query, "format": "json", "engines": "duckduckgo,brave"})
    url = SEARXNG_URL + "/search?" + params
    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=8) as r:
                data = json.loads(r.read().decode())
            results = data.get("results", [])[:max_results]
            if not results:
                log.debug("[search] no results for: %s", query)
                return ""
            lines = []
            for i, r in enumerate(results, 1):
                title   = r.get("title", "")
                content = r.get("content", "")[:300]
                url_    = r.get("url", "")
                lines.append(f"[{i}] {title}\n{content}\n{url_}")
            return "\n\n".join(lines)
        except Exception as e:
            last_err = e
            log.warning("[search] attempt %d failed: %s", attempt + 1, e)
            if attempt < retries:
                time.sleep(2 ** attempt)
    log.error("[search] all retries failed: %s", last_err)
    return ""

def smart_search(query, history=None, max_results=5):
    return tool_search_multi(query, max_results=max_results)
