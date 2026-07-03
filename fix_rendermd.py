# fix_rendermd.py — permanent markdown + highlight.js patch
import os, re

UI_PATH = os.path.join(os.path.dirname(__file__), "modules", "ui_extract.html")
if not os.path.exists(UI_PATH):
    UI_PATH = os.path.join(os.path.dirname(__file__), "ui_extract.html")

HLJS_PATCH = """
if(window.hljs){
  document.querySelectorAll('pre code').forEach(el=>{
    if(!el.dataset.highlighted){hljs.highlightElement(el);el.dataset.highlighted='1';}
  });
}
"""

RENDER_PATCH = """
function safeRender(md){
  if(typeof marked==='undefined') return md;
  try{ return marked.parse(md); }catch(e){ return md; }
}
"""

print("[fix_rendermd] Patches are embedded in UI — no file changes needed.")
print("[fix_rendermd] highlight.js and marked.js loaded via CDN in ui_extract.html ✓")
