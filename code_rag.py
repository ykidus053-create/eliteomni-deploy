import os, re, math
from collections import Counter

def _tokenize(text):
    return re.findall(r'\b\w{3,}\b', text.lower())

def _tf(tokens):
    count = Counter(tokens)
    total = len(tokens)
    return {word: count[word] / total for word in count} if total > 0 else {}

def _cosine_sim(vec1, vec2):
    if not vec1 or not vec2: return 0.0
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    return numerator / (math.sqrt(sum1) * math.sqrt(sum2)) if sum1 and sum2 else 0.0

def _chunk_code(code: str, chunk_size: int = 20) -> list:
    lines = code.split('\n')
    return ['\n'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

def get_relevant_code_context(query: str, top_k: int = 3) -> str:
    """Scans all .py files in the project, chunks them, and returns the most relevant code snippets."""
    snippets = []
    try:
        for root, _, files in os.walk('.'):
            if any(ex in root for ex in ['.git', '__pycache__', 'venv', 'node_modules']): continue
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    for chunk in _chunk_code(code):
                        if len(chunk.strip()) > 20:
                            snippets.append((filepath, chunk))
                            
        if not snippets: return ""
        
        query_vec = _tf(_tokenize(query))
        scored = []
        for filepath, chunk in snippets:
            chunk_vec = _tf(_tokenize(chunk))
            sim = _cosine_sim(query_vec, chunk_vec)
            if sim > 0.05:
                scored.append((sim, filepath, chunk))
                
        scored.sort(key=lambda x: -x[0])
        
        if not scored: return ""
        
        context = ["[RELEVANT CODEBASE CONTEXT]"]
        for sim, filepath, chunk in scored[:top_k]:
            context.append(f"--- {filepath} (Score: {sim:.2f}) ---\n{chunk}\n")
        context.append("[END CODEBASE CONTEXT]")
        return "\n".join(context)
    except Exception:
        return ""
