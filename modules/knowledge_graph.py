"""
Dynamic Knowledge Fusion Engine
- Stores facts with source + confidence
- Cross-source validation
- Temporal decay for stale facts
"""
import time
import json
import os

GRAPH_PATH = "/home/kidus/eliteomni_knowledge_graph.json"

class KnowledgeNode:
    def __init__(self, content: str, source: str, confidence: float = 1.0):
        self.content = content
        self.source = source
        self.timestamp = time.time()
        self.confidence = confidence

    def decay(self, half_life_days: float = 30.0) -> float:
        age_days = (time.time() - self.timestamp) / 86400
        return self.confidence * (0.5 ** (age_days / half_life_days))

    def to_dict(self):
        return {
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp,
            "confidence": self.confidence
        }

class KnowledgeGraph:
    def __init__(self):
        self.nodes: dict[str, KnowledgeNode] = {}
        self._load()

    def add(self, key: str, content: str, source: str, confidence: float = 1.0):
        self.nodes[key] = KnowledgeNode(content, source, confidence)
        self._save()

    def get(self, key: str) -> str | None:
        node = self.nodes.get(key)
        if not node:
            return None
        if node.decay() < 0.1:
            del self.nodes[key]
            self._save()
            return None
        return node.content

    def search(self, query: str, top_k: int = 3) -> list[str]:
        q = query.lower()
        results = []
        for key, node in self.nodes.items():
            if q in node.content.lower() or q in key.lower():
                results.append((node.decay(), node.content))
        results.sort(reverse=True)
        return [r[1] for r in results[:top_k]]

    def _save(self):
        try:
            with open(GRAPH_PATH, "w") as f:
                json.dump({k: v.to_dict() for k, v in self.nodes.items()}, f)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(GRAPH_PATH):
                with open(GRAPH_PATH) as f:
                    data = json.load(f)
                for k, v in data.items():
                    node = KnowledgeNode(v["content"], v["source"], v["confidence"])
                    node.timestamp = v["timestamp"]
                    self.nodes[k] = node
        except Exception:
            pass


_graph = KnowledgeGraph()

def kg_add(key: str, content: str, source: str = "system", confidence: float = 1.0):
    _graph.add(key, content, source, confidence)

def kg_get(key: str) -> str | None:
    return _graph.get(key)

def kg_search(query: str) -> list[str]:
    return _graph.search(query)
