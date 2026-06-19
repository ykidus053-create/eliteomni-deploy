import re, json
from typing import Dict, List, Callable

_SYSTEM = ('You are a reasoning planner. Analyze the request and output JSON only: '
           '{"true_intent": str, "ambiguities": [str], "strategy": str, '
           '"structure": [str], "risks": [str], "confidence": float, '
           '"needs_search": bool, "needs_calculation": bool}')

def analyze_intent(msg: str, generate_fn: Callable,
                   skill: str = 'general', complexity: str = 'medium') -> Dict:
    if complexity == 'easy':
        return {'true_intent': msg, 'ambiguities': [], 'strategy': 'direct',
                'structure': [], 'risks': [], 'confidence': 0.9,
                'needs_search': False, 'needs_calculation': False}
    prompt = [{'role': 'system', 'content': _SYSTEM},
              {'role': 'user', 'content': f'Analyze: {msg[:600]}'}]
    try:
        raw = generate_fn(prompt)
        if raw:
            raw = re.sub(r'```json|```', '', raw).strip()
            s, e = raw.find('{'), raw.rfind('}') + 1
            if s >= 0 and e > s: return json.loads(raw[s:e])
    except: pass
    return _fallback(msg)

def _fallback(msg: str) -> Dict:
    ml = msg.lower()
    return {'true_intent': msg, 'ambiguities': [], 'risks': [],
            'strategy': 'step_by_step' if any(w in ml for w in ['how','explain','why']) else 'direct',
            'structure': [], 'confidence': 0.7,
            'needs_search': any(w in ml for w in ['latest','current','today','news']),
            'needs_calculation': any(w in ml for w in ['calculate','compute','how much'])}

def build_intent_aware_system(base_system: str, intent: Dict,
                              skill: str, complexity: str) -> str:
    addons = []
    if intent.get('ambiguities'):
        addons.append(f'[AMBIGUITY: {intent["ambiguities"][0]} -- address this]')
    strategy = intent.get('strategy', 'direct')
    if strategy == 'step_by_step': addons.append('[STRATEGY: number each step clearly]')
    elif strategy == 'comparative': addons.append('[STRATEGY: use table or parallel structure]')
    if intent.get('structure'):
        addons.append(f'[STRUCTURE: {" -> ".join(intent["structure"][:4])}]')
    if intent.get('confidence', 1.0) < 0.6:
        addons.append('[LOW CONFIDENCE: hedge key claims, acknowledge uncertainty]')
    if not addons: return base_system
    return base_system + '\n\n' + '\n'.join(addons)

def should_run_pregen(msg: str, skill: str, complexity: str) -> bool:
    if complexity == 'easy': return False
    if skill in ('coder', 'researcher') and complexity in ('medium', 'hard'): return True
    if complexity == 'hard': return True
    triggers = ['explain','why','how does','compare','analyze','design','help me']
    return any(w in msg.lower() for w in triggers) and len(msg) > 50
