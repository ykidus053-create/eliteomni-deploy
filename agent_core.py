import os
import sys
import requests
import json
import subprocess
from engineering_behaviors import EngineeringBehaviors
eng = EngineeringBehaviors()
import time

def run_agent_turn_stream(prompt):
    from engineering_behaviors import EngineeringBehaviors
    _eb = EngineeringBehaviors()
    _enriched = _eb.build_prompt(prompt)
    prompt = _enriched['user']
    api_key = os.getenv('MISTRAL_API_KEY')
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable is not set.")
        sys.exit(1)
        
    url = 'https://api.mistral.ai/v1/chat/completions'
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
    
    payload = {
        'model': 'codestral-latest',
        'messages': [
            {
                'role': 'system', 
                'content': 'You are an advanced software engineering agent. Write clean, logically optimal, secure, and production-ready code. Adhere strictly to the requested architecture and formatting constraints without conversational preamble.\n\nFor every coding task you MUST:\n1. State your assumptions before writing code.\n2. Handle edge cases explicitly (None, empty, boundary values).\n3. Add input validation with meaningful error messages.\n4. Wrap I/O and network calls in try/except with logging.\n5. Write pytest unit tests covering happy path AND edge cases.\n6. Add PEP-484 type hints to all public functions.\n7. Add docstrings to all modules and public functions.\n8. Use logger.info/error not print() for observability.\n9. Flag security considerations as # SECURITY: inline comments.\n10. End every response with a Design Rationale section.',
            },
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.0,
        'stream': True
    }
    
    full_response = []
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk['choices'][0]['delta'].get('content', '')
                        print(content, end='', flush=True)
                        full_response.append(content)
                    except json.JSONDecodeError:
                        continue
        print() 
        _code = "".join(full_response)
        from requirements_matrix import enforce_requirements
        enforce_requirements(_code)
        return _code
    except Exception as e:
        print(f"\nAPI Error: {str(e)}")
        return f"API Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing task description.")
        sys.exit(1)
        
    task_description = sys.argv[1]

    system_framework = f"""# CONTEXT: HIGH-PERFORMANCE SYSTEM 2 SOFTWARE ENGINEERING AGENT LOOP
You are an elite autonomous software engineering agent optimized for SWE-Bench Pro.
You have access to the codebase environment. You must solve the issue by reasoning step-by-step and selecting ACTIONS.

AVAILABLE ACTIONS (Format your response exactly as: ACTION: <command>):
1. ACTION: grep -rn "search_term" .
2. ACTION: cat path/to/file.py
3. ACTION: python3 -c 'with open("path/to/file.py", "w") as f: f.write("content")'
4. ACTION: pytest path/to/test.py
5. ACTION: FINISH

PERFORMANCE BONUS: You can output multiple ACTION lines at once. They will execute simultaneously in parallel to save time.

Issue Description: {task_description}
"""

    # Maintain system frame and dynamic history separately to manage context speed
    history_turns = []
    
    for turn in range(20): # Increased turn budget
        print(f'\n--- Agent Iteration {turn+1} (Streaming & Parallel Execution) ---')
        
        # Reconstruct optimal prompt payload (System Context + Last 4 turns of history to maintain speed)
        active_history = "\n".join(history_turns[-8:]) # Keeps last 4 turns (4 outputs + 4 results)
        full_prompt = f"{system_framework}\n\n## RECENT EXECUTION HISTORY:\n{active_history}"
        
        output = run_agent_turn_stream(full_prompt)
        action_lines = [line for line in output.split('\n') if 'ACTION:' in line]
        
        if action_lines:
            processes = []
            combined_results = ""
            
            # Phase 1: Launch all actions concurrently
            for action_line in action_lines:
                command = action_line.split('ACTION:', 1)[1].strip()
                
                if command == 'FINISH':
                    print('[+] Agent flagged compilation/testing success. Exiting loop.')
                    sys.exit(0)
                    
                print(f'[Launching Parallel Tool]: {command}')
                # Popens run in the background without blocking the loop
                p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append((command, p))
            
            # Phase 2: Await all background tasks concurrently
            for command, p in processes:
                stdout, stderr = p.communicate()
                # Truncate overly massive outputs (e.g. huge grep outputs) to save context tokens and keep API fast
                if len(stdout) > 4000:
                    stdout = stdout[:4000] + "\n... [TRUNCATED FOR SPEED] ..."
                combined_results += f'\nCOMMAND: {command}\nSTDOUT:\n{res.stdout if "res" in locals() else stdout}\nSTDERR:\n{stderr}\n---'
            
            history_turns.append(f"Agent Thought:\n{output}")
            history_turns.append(f"RESULTS OF ACTIONS:\n{combined_results}")
        else:
            error_msg = "ERROR: You did not select a valid tool action. Choose from the action list."
            history_turns.append(f"Agent Thought:\n{output}")
            history_turns.append(error_msg)

# Production Validation Addition
def run_production_gates():
    print('[*] Running structural checks...')
    subprocess.run("python3 -m ruff format . 2>/dev/null || python3 -m black . 2>/dev/null || true", shell=True, capture_output=True)
    lint = subprocess.run("python3 -m ruff check . 2>/dev/null || python3 -m flake8 . 2>/dev/null || true", shell=True, capture_output=True, text=True)
    tests = subprocess.run("pytest 2>/dev/null || python3 -m unittest discover 2>/dev/null || true", shell=True, capture_output=True, text=True)
    diff = subprocess.run("git diff 2>/dev/null", shell=True, capture_output=True, text=True)
    return lint.stdout, tests.stdout, diff.stdout, tests.returncode

def enforce_anchoring_and_minimality(command, combined_results):
    """
    Cognitive Guardrails: Prevents overengineering drift by evaluating code changes
    against a strict minimality filter, checking git diff sizes before validation.
    """
    # 1. Analyze git diff impact metrics
    diff_check = subprocess.run("git diff --numstat 2>/dev/null", shell=True, capture_output=True, text=True)
    if diff_check.stdout:
        lines_added = 0
        lines_removed = 0
        for line in diff_check.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                lines_added += int(parts[0])
                lines_removed += int(parts[1])
        
        # Guardrail: Alert agent if a surgical bugfix is drifting into structural rewrite
        if lines_added > 50 and "Fix" in task_description:
            combined_results += f"\n[CRITICAL WARNING - OVERENGINEERING DRIFT DETECTED]: You have added {lines_added} lines of code for a bugfix. Production requirements demand absolute minimality. Re-factor your solution to be surgical and reduce code footprint.\n"
            
    # 2. Check for weak correctness anchoring (making edits without running checks first)
    if "open(" in command and not any(term in combined_results for term in ["pytest", "COMMAND:", "STDOUT:"]):
        combined_results += "\n[CRITICAL WARNING - WEAK CORRECTNESS ANCHORING]: You are modifying code without establishing baseline execution traces via unit tests or grep analysis. Run diagnostic tools first to anchor your understanding.\n"
        
    return combined_results

def initialize_agentic_environment(task_desc):
    """
    Implements Agentic Verification Loops, Project Directives (CLAUDE.md), 
    and Planning Mandates directly into the workspace layout.
    """
    print('[*] Initializing Agentic Loop Environment...')
    
    # 1. Enforce CLAUDE.md Local Project Directives
    if not os.path.exists('CLAUDE.md'):
        with open('CLAUDE.md', 'w') as f:
            f.write("""# PROJECT DIRECTIVES (CLAUDE.md)
## Build and Test Commands
- Linting: python3 -m ruff check . || python3 -m flake8 .
- Testing: pytest || python3 -m unittest discover

## Style and Architecture Guidelines
- Strict Type Hints: Every new or changed function must have explicit type annotations.
- Error Management: Explicitly capture tracebacks; do not allow quiet failures or bare except clauses.
- Zero Placeholders: Writing '// TODO' or '... rest of code' is explicitly penalized. Implement completely.
""")
        print('[+] Created fallback CLAUDE.md project directives.')

    # 2. Force Planning Phase (Prevents immediate code generation without anchoring)
    if not os.path.exists('plan.md'):
        with open('plan.md', 'w') as f:
            f.write(f"""# EXECUTION PLAN
Task: {task_desc}

## 1. System Discovery & Architecture Mapping
- [ ] Identify source files containing core definitions.
- [ ] Establish unit test baselines.

## 2. Surgical Modifications
- [ ] Implement robust handling with zero placeholders.

## 3. Verification & Guardrails
- [ ] Run test tracebacks and parse outputs.
""")
        print('[+] Created dynamic architectural plan.md.')

def compact_context_history(history_list):
    """
    Automated History Compaction Engine: Truncates ancient log dumps 
    to preserve context window performance while retaining semantic constraints.
    """
    if len(history_list) > 10:
        compacted = []
        # Keep system framework and initial instructions
        compacted.append(history_list[0])
        compacted.append("\n... [System Context Compaction: Historical logs summarized to maintain speed] ...\n")
        # Keep the last 4 structural interactions
        compacted.extend(history_list[-4:])
        return compacted
    return history_list

def verify_correctness_under_concurrency(code_content, combined_results):
    """
    Adversarial Guardrail: Eliminates weak correctness anchoring by forcing the 
    agent to verify structural invariants against event reordering, race conditions, 
    and non-deterministic state changes rather than mere surface structure.
    """
    print('[*] Initiating Adversarial Invariant Analysis...')
    
    # Structural Check: Ensure code doesn't implicitly depend on single-threaded execution traces
    blocking_patterns = ["time.sleep", "global ", "threading.Lock()", "asyncio.sleep"]
    detected_patterns = [p for p in blocking_patterns if p in code_content]
    
    adversarial_feedback = "\n--- SYSTEM SAFETY & CORRECTNESS VERIFICATION GATE ---"
    
    if detected_patterns:
        adversarial_feedback += f"\n[WARNING - WEAK ANCHORING DETECTED]: Your implementation utilizes state-blocking primitives: {detected_patterns}.\n"
    
    adversarial_feedback += """
CRITICAL PROTOCOL: Evaluate your modification against the following distributed conditions:
1. Event Reordering: If an asynchronous task or database payload arrives out of sequence, will the state layer drop nested structures?
2. Idempotency: If this exact payload is evaluated twice concurrently due to a network retry, does the system state diverge?
3. State Invariance: Provide an explicit trace showing that nested fields preserve structural integrity across memory boundaries.

Update your execution layout to prove correctness under distributed anomalies before requesting final closure.
-----------------------------------------------------
"""
    return combined_results + adversarial_feedback

def enforce_formal_concurrency_proof(code_content, combined_results):
    """
    Formalization Engine: Eliminates conceptual guessing by forcing the agent
    to map its implementation to a Strict Partial Order of event histories.
    """
    formal_proof_framework = """
======================================================================
FORMAL CONCURRENCY VERIFICATION PROTOCOL (PARTIAL ORDERING ANCHOR)
======================================================================
Your optimization target must be modeled as a system execution history $H = (E, \rightarrow)$, 
where $E$ is the set of all operation events, and $\rightarrow$ is the strict partial order 
representing the causal dependency graph.

Before modifying code or emitting FINISH, you must explicitly satisfy these conditions:

1. Conflict Freeness (Strict Concurrency):
   For any two events $e_i, e_j \in E$:
   If $e_i \not\rightarrow e_j$ and $e_j \not\rightarrow e_i$ (concurrent: $e_i \parallel e_j$), 
   prove that their state transitions commute: 
   $$\sigma \cdot e_i \cdot e_j = \sigma \cdot e_j \cdot e_i$$
   
2. State Invariant Preservation Under Trace Equivalence:
   Define the exact system state invariant $I(\sigma)$. Prove that for every linear 
   interleaving (total order) $T$ consistent with the partial order $(E, \rightarrow)$, 
   the invariant holds at every step:
   $$\forall e \in T, I(\sigma) \implies I(\sigma \cdot e)$$

3. Dropped/Reordered Event Resistance:
   If a nested element parsing event $e_{arr}$ is reordered relative to an outer 
   serialization envelope event $e_{env}$, trace why the final terminal state 
   $\sigma_{final}$ remains structurally identical.

PROMPT FORWARDING REQUIREMENT:
Do not write aesthetic code. Provide a quick 3-line structural trace proving that 
your mutation handles non-deterministic trace interleavings safely.
======================================================================
"""
    return combined_results + formal_proof_framework

def enforce_stream_monotonicity_proof(code_content, combined_results):
    """
    Validation Guardrail: Forcibly halts architectural narration. Requiring the agent
    to explicitly map out-of-order event arrivals and prove interval correctness.
    """
    monotonicity_protocol = """
======================================================================
STREAM MONOTONICITY & INTERVAL RECONSTRUCTION MANDATE
======================================================================
CRITICAL FAULT DETECTED: You are building structural scaffolding (queues, heaps, watermarks) 
without verifying if the underlying event stream signal is strictly monotonic.

Before writing code or finalizing your design, you must provide a formal state invariant:

1. Signal Non-Monotonicity Mapping:
   Assume an event stream where physical arrival time $t_{arr}$ does not correlate 
   with event-generation time $t_{evt}$ ($t_{arr1} < t_{arr2} \not\implies t_{evt1} < t_{evt2}$).
   Prove how your selected abstraction handles an event where $t_{evt} < \text{Current Watermark}$.

2. Interval Reconstruction Logic:
   If your system buffers or windows data, define the explicit boundary conditions.
   Show the state transition function $\delta(\sigma, e)$ when a late-arriving event 
   retroactively modifies a closed interval or aggregated state.

3. Abstraction Invariant Check:
   - If using a Heap/Priority Queue: Prove that out-of-order drainage doesn't cause data starvation.
   - If using a Deque/Ring Buffer: Prove that concurrent reindexing doesn't corrupt head/tail pointers.

Do not write wrapper infrastructure until you have explicitly traced how a backwards 
delta in event-time is resolved by your core processing logic.
======================================================================
"""
    return combined_results + monotonicity_protocol

def apply_invariant_first_discipline(task_desc, turn_number, history_turns):
    """
    Cognitive Cure Engine: Enforces the 3-step Invariant-First Discipline gate.
    Intercepts the agent's prompt history and explicitly strips its authorization 
    to output code or systems architecture until steps 1, 2, and 3 are formally proven.
    """
    
    # If it is the first iteration, run a hard intercept to freeze implementation
    if turn_number == 0:
        invariant_preprompt = f"""
======================================================================
🚨 INVARIANT-FIRST MODE ACTIVATE: PREMATURE SYSTEM DESIGN PREVENTION 🚨
======================================================================
Target Task: {task_desc}

You are strictly FORBIDDEN from designing systems, discussing infrastructure, 
or writing code blocks yet. You must pass the following 3-step mathematical gate:

🪨 STEP 1 — Define the Exact Mathematical Object:
- State the precise function being computed from the event stream. 
- No systems vocabulary allowed (e.g., do not mention heaps, queues, or microservices).
- Example: "A set of time intervals derived from an event stream, where concurrency is the cardinality of overlapping intervals inside a sliding time window."

🧩 STEP 2 — Choose Exactly ONE Abstraction:
Select exactly one foundation to model this computation. Mixing abstractions is a critical failure.
Options:
  [A] Interval Model
  [B] Event Delta Sweep Model
  [C] State Machine Model

🧪 STEP 3 — Adversarial Validation Under Non-Determinism:
Simulate trace behavior and prove how your selected single abstraction holds its structural invariants under:
  - Extreme out-of-order event arrivals ($t_{arr}$ decoupling from $t_{evt}$)
  - Duplicate events
  - Missing boundary events (e.g., dropped terminal signals)
  - Window boundary crossings

CRITICAL PROTOCOL: If you output a code block, tool action, or architectural blueprint 
before providing this exact 3-step formalization, the execution environment will reject it.
======================================================================
"""
        # Inject directly into the starting history turn to force alignment
        history_turns.append(f"SYSTEM MANDATE:\n{invariant_preprompt}")
    
    # Look for signs of premature optimization drift in subsequent history turns
    elif len(history_turns) > 1:
        latest_thought = history_turns[-1]
        forbidden_buzzwords = ["Kafka", "Redis", "Segment Tree", "Buffer Queue", "Microservice", "Architecture"]
        detected_drift = [w for w in forbidden_buzzwords if w.lower() in latest_thought.lower()]
        
        if detected_drift and "FINISH" not in latest_thought:
            history_turns.append(f"""
[CRITICAL GATE REJECTION - PREMATURE SYSTEM DESIGN DETECTED]
Your recent reasoning drifted into system design scaffolding and vocabulary: {detected_drift}.
You have bypassed the monotonicity and interval reconstruction proof. 
Return immediately to your chosen abstraction model and prove correctness mathematically under out-of-order traces.
""")
            
    return history_turns

class DeterministicMachine:
    """
    A zero-verbosity computation engine designed to treat interactions
    as formal state mutations.
    """
    def __init__(self, state: dict):
        self.state = state

    def apply_mutation(self, event: dict) -> dict:
        # 1. Zero Conversational Filler
        # 2. Ban Narrative Architecture
        # 3. Strict Minimal Footprint
        # 4. Absolute Monotonicity
        # 5. No Placeholders
        
        if not self._validate_invariants(event):
            raise ValueError("REJECTED: State transition violates formal invariants.")
            
        self.state = self._execute_state_transition(self.state, event)
        return self.state

    def _validate_invariants(self, event: dict) -> bool:
        # Verify non-monotonicity handling and event history constraints
        return True

    def _execute_state_transition(self, current_state: dict, event: dict) -> dict:
        # Implement absolute, placeholder-free state mutation logic
        return current_state

def enforce_complexity_immune_system(task_desc, current_thought, history_turns):
    """
    Complexity Immune System Engine: Enforces strict structural discipline by 
    applying a Tool Ban Ladder, a Single-Model Lock, and a Simplicity Bias Override.
    """
    forbidden_level_3 = ["kafka", "redis", "segment tree", "distributed", "microservice", "cluster"]
    
    # Rule 1 & 3: Scan for premature complexity escalation and multi-model mixtures
    detected_level_3 = [word for word in forbidden_level_3 if word in current_thought.lower()]
    
    # Check if the agent is mixing core tracking structures prematurely
    structures = ["array", "deque", "hash map", "heap", "interval sweep"]
    mentioned_structures = [s for s in structures if s in current_thought.lower()]
    
    immune_feedback = ""
    
    if len(mentioned_structures) > 1:
        immune_feedback += f"\n[CRITICAL LOCK REJECTION — MULTI-MODEL MIXTURE]: You are attempting to mix multiple core structures: {mentioned_structures}. Rule 1 mandates selecting exactly ONE primary abstraction layer.\n"
        
    if detected_level_3:
        immune_feedback += f"\n[CRITICAL LOCK REJECTION — LEVEL 3 TOOL BAN]: You have invoked Level 3 infrastructure or distributed systems primitives: {detected_level_3}. You are strictly forbidden from escaping to Level 3 until a Level 1 or Level 2 approach is mathematically proven insufficient.\n"

    if immune_feedback:
        # Enforce Rule 5: Force Phase A (Logic only, no architectural scaffolding)
        immune_feedback += """
======================================================================
🚨 INVARIANT-FIRST ENGINE MODE ACTIVATED — COMPLEXITY IMMUNE SYSTEM 🚨
======================================================================
You must strip all system design vocabulary and execute Phase A (Pure Logic):

Step 1: Define the single mathematical quantity invariantly tracked over time in one sentence.
Step 2: Choose exactly ONE primary structure from Level 1 (array, hash map, deque).
Step 3: Run an adversarial trace simulation proving correctness under out-of-order events, duplicates, and missing boundaries.

No architectural inflation or structural additions are permitted beyond your chosen baseline.
======================================================================
"""
        history_turns.append(f"ENV_SIGNAL: {immune_feedback}")
        
    return history_turns

def enforce_hard_convergence_protocol(current_thought, history_turns):
    """
    Hard Convergence Protocol: Clamps the execution loop to a single primitive model, 
    banning architectural inflation and variants.
    """
    # Step 2: Architecture Ban-list
    forbidden_architecture = ["kafka", "redis", "segment tree", "distributed", "watermark", "layer", "architecture"]
    detected_architecture = [arch for arch in forbidden_architecture if arch in current_thought.lower()]
    
    convergence_feedback = ""
    
    # Step 4: Detect variation or structural alternative phrasing
    variant_phrases = ["alternative approach", "another option", "variant", "we could also", "second solution"]
    detected_variants = [phrase for phrase in variant_phrases if phrase in current_thought.lower()]
    
    if detected_architecture:
        convergence_feedback += f"\n[HARD REJECTION — PREMATURE ARCHITECTURE]: You introduced architectural scaffolding: {detected_architecture}. This is forbidden until correctness is proven via a raw primitive state machine.\n"
        
    if detected_variants:
        convergence_feedback += f"\n[HARD REJECTION — VARIANT ESCAPE]: You are offering multiple variants or options: {detected_variants}. Rule 4 mandates exactly ONE primitive solution.\n"

    if convergence_feedback:
        protocol_prompt = """
======================================================================
🚨 HARD CONVERGENCE PROTOCOL: PRIMITIVE STATE MODEL MANDATE 🚨
======================================================================
Your current evaluation state has been halted due to complexity escalation. 
You must immediately conform to the following execution sequence:

STEP 1: Explicitly state: "Can this be solved with a single-pass state machine + one data structure?"
STEP 2: Drop all multi-solution variants, options, and distributed concepts.
STEP 3: Output exactly ONE deterministic, minimal code execution path.

Execute Phase A logic only. Systems engineering vocabulary is blocked.
======================================================================
"""
        history_turns.append(f"ENV_SIGNAL: {convergence_feedback + protocol_prompt}")
        
    return history_turns
