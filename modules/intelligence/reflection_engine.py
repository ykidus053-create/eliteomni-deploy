import re, time, json, sqlite3
from pathlib import Path
from typing import List, Tuple

DB = Path.home() / "eliteomni_reflections.db"

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS reflections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT, msg_preview TEXT, response_preview TEXT,
        skill TEXT, critique TEXT, revision_needed INTEGER,
        improvement_applied INTEGER DEFAULT 0, ts REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS reflection_patterns (
        pattern TEXT PRIMARY KEY, frequency INTEGER,
        skill TEXT, last_seen REAL)""")
    c.commit()
    return c

def critique_response(response: str, msg: str, skill: str) -> List[Tuple[str, str, float]]:
    issues = []
    r = response
    q_count = msg.count("?")
    if q_count > 1:
        paragraphs = [p for p in r.split("\n\n") if p.strip()]
        if len(paragraphs) < q_count:
            issues.append(("completeness",
                           f"Question has {q_count} parts but response has only {len(paragraphs)} paragraphs",
                           0.7))
    if re.match(r"^(Certainly|Sure|Of course|Absolutely|Great question|I would be happy|I am happy)", r, re.I):
        issues.append(("directness", "Response opens with filler — delete first sentence", 0.5))
    vague_count = len(re.findall(r"\b(various|several|many things|some cases|often|usually|typically)\b", r, re.I))
    if vague_count > 3:
        issues.append(("specificity", f"{vague_count} vague quantifiers — add concrete examples", 0.4 + vague_count * 0.05))
    sentences = re.split(r"(?<=[.!?])\s+", r)
    negations = [s for s in sentences if re.search(r"\b(not|never|no longer|impossible)\b", s, re.I)]
    positives = [s for s in sentences if re.search(r"\b(always|definitely|certainly|must)\b", s, re.I)]
    if negations and positives and len(sentences) > 4:
        for neg in negations[:2]:
            neg_topic = re.findall(r"\b\w{5,}\b", neg)[:3]
            for pos in positives[:2]:
                if any(t in pos for t in neg_topic):
                    issues.append(("consistency", "Possible contradiction: negation and absolute claim on same topic", 0.6))
                    break
    msg_len = len(msg.split())
    resp_len = len(r.split())
    if msg_len < 10 and resp_len > 300:
        issues.append(("calibration", f"Short question ({msg_len}w) got very long response ({resp_len}w)", 0.3))
    elif msg_len > 50 and resp_len < 50:
        issues.append(("calibration", f"Complex question ({msg_len}w) got very short response ({resp_len}w)", 0.5))
    return sorted(issues, key=lambda x: x[2], reverse=True)[:4]

def build_reflection_injection(msg: str, skill: str) -> str:
    reflections = []
    try:
        c = _conn()
        rows = c.execute(
            "SELECT pattern FROM reflection_patterns WHERE skill=? AND frequency >= 2 ORDER BY frequency DESC LIMIT 3",
            (skill,)).fetchall()
        c.close()
        reflections = [r[0] for r in rows]
    except: pass
    parts = ["\n<pre_generation_reflection>"]
    parts.append(f"Before generating, internalize these quality checks for {skill} tasks:")
    parts.append("1. Lead with the direct answer, not preamble")
    parts.append("2. Match response length to question complexity")
    parts.append("3. Replace vague quantifiers with specific evidence")
    parts.append("4. Every claim needs a basis — mark uncertain claims explicitly")
    if reflections:
        parts.append("Known failure patterns to avoid:")
        for r in reflections:
            parts.append(f"  - {r}")
    parts.append("</pre_generation_reflection>")
    return "\n".join(parts)

def store_reflection(session: str, msg: str, response: str, skill: str,
                     issues: List[Tuple]):
    if not issues: return
    try:
        c = _conn()
        critique_text = " | ".join(f"{i[0]}: {i[1]}" for i in issues)
        c.execute("INSERT INTO reflections VALUES (NULL,?,?,?,?,?,0,0,?)",
            (session, msg[:100], response[:200], skill, critique_text, time.time()))
        for dim, issue, severity in issues:
            pattern = f"{dim}: {issue[:60]}"
            row = c.execute("SELECT frequency FROM reflection_patterns WHERE pattern=?", (pattern,)).fetchone()
            if row:
                c.execute("UPDATE reflection_patterns SET frequency=?, last_seen=? WHERE pattern=?",
                          (row[0]+1, time.time(), pattern))
            else:
                c.execute("INSERT INTO reflection_patterns VALUES (?,1,?,?)",
                          (pattern, skill, time.time()))
        c.commit(); c.close()
    except Exception as e:
        print(f"[Reflection] {e}")
