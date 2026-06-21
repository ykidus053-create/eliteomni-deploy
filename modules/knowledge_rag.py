import os, json, time, math, hashlib, sqlite3
DB_PATH = os.environ.get("MEMORY_DB", "/home/kidus/eliteomni_memory.db")

def _conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS rag_docs (id INTEGER PRIMARY KEY, doc_id TEXT, chunk TEXT, source TEXT, ts REAL, emb TEXT)")
    con.commit()
    return con

def chunk_text(text, size=400, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def embed(texts):
    try:
        import requests
        key = os.environ.get("MISTRAL_API_KEY", "")
        r = requests.post("https://api.mistral.ai/v1/embeddings", headers={"Authorization": "Bearer "+key, "Content-Type": "application/json"}, json={"model": "mistral-embed", "inputs": texts}, timeout=30)
        if r.status_code == 200: return [d["embedding"] for d in r.json()["data"]]
    except Exception as e: print("[rag] embed error:", e)
    vocab = {}
    for t in texts:
        for w in t.lower().split():
            if w not in vocab: vocab[w] = len(vocab)
    def bow(t):
        v = [0.0]*len(vocab)
        for w in t.lower().split():
            if w in vocab: v[vocab[w]] += 1.0
        norm = math.sqrt(sum(x*x for x in v)) or 1.0
        return [x/norm for x in v]
    return [bow(t) for t in texts]

def cosine(a, b):
    if len(a) != len(b): return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb+1e-9)

def ingest(text, source="user"):
    chunks = chunk_text(text)
    if not chunks: return 0
    embs = embed(chunks)
    doc_id = hashlib.md5(text[:200].encode()).hexdigest()
    with _conn() as con:
        for chunk, emb in zip(chunks, embs):
            con.execute("INSERT INTO rag_docs (doc_id,chunk,source,ts,emb) VALUES (?,?,?,?,?)", (doc_id, chunk, source, time.time(), json.dumps(emb)))
    print("[rag] ingested", len(chunks), "chunks from", source)
    return len(chunks)

def _bm25(qterms, dtoks, avgdl, N, df):
    k1, b = 1.5, 0.75
    tf = {}
    for tok in dtoks: tf[tok] = tf.get(tok,0)+1
    score = 0.0
    for term in qterms:
        idf = math.log((N-df.get(term,0)+0.5)/(df.get(term,0)+0.5)+1)
        tfn = (tf.get(term,0)*(k1+1))/(tf.get(term,0)+k1*(1-b+b*len(dtoks)/max(avgdl,1)))
        score += idf*tfn
    return score

def retrieve(query, top_k=5, history=None):
    try:
        from modules.search import rewrite_query
        query = rewrite_query(query, history)
    except Exception: pass
    with _conn() as con:
        rows = con.execute("SELECT chunk, emb FROM rag_docs").fetchall()
    if not rows: return []
    chunks = [r[0] for r in rows]
    embs   = [json.loads(r[1]) for r in rows]
    qemb   = embed([query])[0]
    tok    = lambda t: t.lower().split()
    qtoks  = tok(query)
    dtoks  = [tok(c) for c in chunks]
    N      = len(chunks)
    avgdl  = sum(len(t) for t in dtoks)/max(N,1)
    df = {}
    for toks in dtoks:
        for t in set(toks): df[t] = df.get(t,0)+1
    bscores = [_bm25(qtoks,dt,avgdl,N,df) for dt in dtoks]
    cscores = [cosine(qemb,e) for e in embs]
    maxb = max(bscores) or 1.0
    ranked = sorted(range(len(chunks)), key=lambda i: 0.5*(bscores[i]/maxb)+0.5*cscores[i], reverse=True)
    return [chunks[i] for i in ranked[:top_k]]

def inject_rag(msgs, query, top_k=5, history=None):
    chunks = retrieve(query, top_k=top_k, history=history)
    if not chunks: return msgs
    ctx = "RETRIEVED CONTEXT:" + chr(10) + (chr(10)+chr(10)).join("["+str(i+1)+"] "+c for i,c in enumerate(chunks))
    result, injected = [], False
    for m in msgs:
        if m.get("role")=="system" and not injected:
            result.append({**m, "content": m["content"]+chr(10)+chr(10)+ctx})
            injected = True
        else:
            result.append(m)
    if not injected: result = [{"role":"system","content":ctx}] + msgs
    print("[rag] injected", len(chunks), "chunks")
    return result

def rag_ask(query, msgs=None, skill="general", max_tokens=2000, top_k=5):
    from modules.guardrails import gateway
    base = (msgs or []) + [{"role":"user","content":query}]
    rag_msgs = inject_rag(base, query, top_k=top_k, history=msgs)
    return gateway(rag_msgs, skill=skill, max_tokens=max_tokens)["response"]