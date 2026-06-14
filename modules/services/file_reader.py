
import os, io

def read_uploaded_file(path: str) -> dict:
    """
    Read CSV, JSON, TXT, PDF, or Excel.
    Returns {"type": ..., "content": ..., "error": ...}
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        try:
            with open(path) as f:
                content = f.read(200000)
            return {"type": "csv", "content": content, "error": None}
        except Exception as e:
            return {"type": "csv", "content": None, "error": str(e)}

    if ext == ".json":
        try:
            import json
            data = json.load(open(path))
            return {"type": "json", "content": json.dumps(data, indent=2)[:50000], "error": None}
        except Exception as e:
            return {"type": "json", "content": None, "error": str(e)}

    if ext in (".txt", ".md", ".log"):
        try:
            content = open(path).read(200000)
            return {"type": "text", "content": content, "error": None}
        except Exception as e:
            return {"type": "text", "content": None, "error": str(e)}

    if ext == ".pdf":
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:20]:
                    text += page.extract_text() or ""
            return {"type": "pdf", "content": text[:100000], "error": None}
        except ImportError:
            try:
                import subprocess, sys
                r = subprocess.run([sys.executable, "-m", "pip", "install",
                                    "pdfplumber", "-q", "--break-system-packages"],
                                   capture_output=True)
                import pdfplumber
                text = ""
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages[:20]:
                        text += page.extract_text() or ""
                return {"type": "pdf", "content": text[:100000], "error": None}
            except Exception as e:
                return {"type": "pdf", "content": None, "error": str(e)}
        except Exception as e:
            return {"type": "pdf", "content": None, "error": str(e)}

    if ext in (".xlsx", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(path)
            return {"type": "excel",
                    "content": df.to_csv(index=False)[:100000],
                    "error": None}
        except Exception as e:
            return {"type": "excel", "content": None, "error": str(e)}

    return {"type": "unknown", "content": None,
            "error": f"Unsupported file type: {ext}"}


def summarize_file(path: str) -> str:
    """Read a file and return a plain-text summary for injection into context."""
    result = read_uploaded_file(path)
    if result["error"]:
        return f"[File Error] {result['error']}"
    content = result["content"] or ""
    ftype = result["type"]

    if ftype == "csv":
        lines = content.split("\n")
        return (f"[CSV File: {os.path.basename(path)}]\n"
                f"Header: {lines[0] if lines else 'unknown'}\n"
                f"Rows: {len(lines)-1}\n"
                f"Preview:\n" + "\n".join(lines[1:4]))

    if ftype in ("pdf", "text"):
        return (f"[{ftype.upper()} File: {os.path.basename(path)}]\n"
                f"{content[:2000]}...")

    return f"[{ftype.upper()} loaded: {len(content)} chars]"
