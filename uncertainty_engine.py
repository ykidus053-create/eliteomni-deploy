
import re

_OVERCONFIDENT = re.compile(
    r"\b(definitely|certainly|always|never|guaranteed|absolutely|"
    r"undeniably|clearly|obviously|without doubt|I am certain)\b",
    re.IGNORECASE)
_HEDGED = re.compile(
    r"\b(may|might|could|probably|likely|approximately|roughly|"
    r"I think|I believe|seems|appears|typically|generally|often|usually)\b",
    re.IGNORECASE)

_REPLACEMENTS = [
    ("I am absolutely certain", "I believe"),
    ("It is definitely", "It is likely"),
    ("This is guaranteed to", "This should"),
    ("will definitely", "will likely"),
    ("always works", "generally works"),
    ("never fails", "rarely fails"),
    ("100% correct", "correct in most cases"),
    ("without any doubt", "with reasonable confidence"),
    ("undeniably", "arguably"),
    ("obviously", "apparently"),
]

def score_response_confidence(text, skill="general"):
    oc = len(_OVERCONFIDENT.findall(text))
    hd = len(_HEDGED.findall(text))
    total = oc + hd + 1
    raw = (oc * 0.9 + hd * 0.5) / total
    return {"confidence": round(min(0.95, max(0.15, raw)), 2), "overconfident": oc, "hedged": hd}

def strip_overconfidence(text):
    for bad, good in _REPLACEMENTS:
        text = re.sub(re.escape(bad), good, text, flags=re.IGNORECASE)
    return text

def inject_confidence_header(text, confidence, skill):
    if confidence >= 0.75 or skill == "general":
        return text
    if confidence < 0.4:
        return "> Low confidence -- please verify independently.\n\n" + text
    return text

def should_hedge(skill, complexity, has_search):
    if skill == "researcher" and not has_search:
        return "\n> Note: My knowledge has a training cutoff. Verify time-sensitive claims."
    return ""

def record_calibration(skill, claim_type, predicted):
    pass
