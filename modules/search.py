import urllib.request, urllib.parse, json, os

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")

def tool_search_multi(query, max_results=5):
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "engines": "duckduckgo,brave"})
        url = SEARXNG_URL + "/search?" + params
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode())
        results = data.get("results", [])[:max_results]
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            url_ = r.get("url", "")
            lines.append("[" + str(i) + "] " + title + "\n" + content + "\n" + url_)
        return "\n\n".join(lines)
    except Exception as e:
        print("[search] error:", e)
        return ""

def smart_search(query, history=None, max_results=5):
    return tool_search_multi(query, max_results=max_results)
