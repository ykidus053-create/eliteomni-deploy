"""
Synthetic self-training loop.
Automatically generates questions → gets AI responses → RLAIF scores → saves to finetune db.
Runs in background continuously. No human needed.
"""
import time, threading, random, os, sys
sys.path.insert(0, os.path.dirname(__file__))

PROMPTS = {
    "coder": [
        "Implement a thread-safe LRU cache with TTL expiry in Python",
        "Write a connection pool manager with retry logic and circuit breaker",
        "Implement a rate limiter using the token bucket algorithm",
        "Write a distributed lock using Redis with deadlock prevention",
        "Implement a WAL (write-ahead log) for crash recovery",
        "Write a producer-consumer queue with backpressure",
        "Implement consistent hashing with virtual nodes",
        "Write a TCP server that handles 10k concurrent connections",
        "Implement a bloom filter with false positive rate control",
        "Write a binary search tree with rebalancing",
        "Implement async retry with exponential backoff and jitter",
        "Write a simple HTTP/1.1 parser from scratch",
        "Implement a merkle tree for data integrity verification",
        "Write a lock-free stack using compare-and-swap",
        "Implement leader election using bully algorithm",
        "Write a pub/sub system with topic filtering",
        "Implement a simple key-value store with MVCC",
        "Write a job scheduler with cron expression parsing",
        "Implement a sliding window rate limiter",
        "Write a stream processor with exactly-once semantics",
    ],
    "general": [
        "Explain gradient descent with a concrete numerical example",
        "What is the difference between TCP and UDP?",
        "Explain CAP theorem with a real-world example",
        "What is the difference between process and thread?",
        "Explain consistent hashing and when to use it",
        "What is the difference between SQL and NoSQL?",
        "Explain the two-phase commit protocol",
        "What is a distributed transaction and how does it work?",
        "Explain the Raft consensus algorithm step by step",
        "What is eventual consistency and when is it acceptable?",
    ],
    "researcher": [
        "What are the tradeoffs between strong and eventual consistency?",
        "Compare Raft vs Paxos consensus algorithms",
        "What are the main failure modes in distributed systems?",
        "How does Google Spanner achieve global consistency?",
        "What is the difference between optimistic and pessimistic locking?",
    ]
}

def _generate_response(prompt: str, skill: str) -> str:
    """Call the local pipeline to get a response."""
    try:
        from modules.core.http_client import mistral_generate
        model = "mistral-code-agent-latest" if skill == "coder" else "magistral-medium-latest"
        msgs = [
            {"role": "system", "content": "You are an expert software engineer. Give complete, production-grade implementations with no stubs or placeholders."},
            {"role": "user", "content": prompt}
        ]
        return mistral_generate(msgs, max_tokens=4000, model=model) or ""
    except Exception as e:
        print(f"[synthetic] generate error: {e}")
        return ""

def _rlaif_score(prompt: str, response: str, skill: str) -> float:
    """AI scores its own response."""
    try:
        from modules.core.http_client import mistral_generate
        import re
        score_msgs = [{"role": "user", "content": f"""Score this response 0.0-1.0:
- 1.0: complete, runnable, no stubs, handles errors properly
- 0.7: mostly complete, minor gaps
- 0.4: has stubs or placeholders
- 0.1: fake or incomplete

Task: {prompt[:200]}
Response: {response[:2000]}

Reply ONLY with a float. Score:"""}]
        result = mistral_generate(score_msgs, max_tokens=10, model="mistral-small-latest") or ""
        match = re.search(r'0\.\d+|1\.0', result)
        return float(match.group()) if match else 0.5
    except:
        return 0.5

def _run_training_loop(iterations: int = 0, delay: float = 10.0):
    """
    Main loop. iterations=0 means run forever.
    delay = seconds between each generation.
    """
    from finetune import finetune_save
    from constitutional_rlaif import critique_and_revise, pairwise_preference
    from modules.core.http_client import mistral_generate

    def _gen(msgs, max_tokens=500):
        return mistral_generate(msgs, max_tokens=max_tokens, model="mistral-small-latest")

    count = 0
    print(f"[synthetic] starting training loop — iterations={'∞' if iterations==0 else iterations}")

    while iterations == 0 or count < iterations:
        try:
            # Pick random skill and prompt
            skill = random.choice(["coder", "coder", "general", "researcher"])  # bias toward coder
            prompt = random.choice(PROMPTS[skill])
            print(f"[synthetic] iter={count+1} skill={skill} prompt='{prompt[:60]}'")

            # Generate response
            response = _generate_response(prompt, skill)
            if not response or len(response) < 50:
                print(f"[synthetic] empty response — skipping")
                time.sleep(delay)
                continue

            # RLAIF score
            score = _rlaif_score(prompt, response, skill)
            print(f"[synthetic] score={score:.2f}")

            # Constitutional critique and revision
            principle, critique, revised = critique_and_revise(prompt, response, _gen)
            print(f"[synthetic] critique: {critique[:60]}")

            # Save pairwise preference if revised is different
            if revised and revised != response and len(revised) > 100:
                pairwise_preference(prompt, revised, response, _gen)

            # Save to finetune db if score is good
            if score >= 0.4:
                finetune_save(skill, "hard" if skill=="coder" else "medium",
                              "You are an expert software engineer.", prompt, response,
                              rating=int(score * 10))
                print(f"[synthetic] saved score={score:.2f}")

            count += 1
            time.sleep(delay)

        except KeyboardInterrupt:
            print(f"[synthetic] stopped after {count} iterations")
            break
        except Exception as e:
            print(f"[synthetic] error: {e}")
            time.sleep(delay * 2)

    print(f"[synthetic] done — {count} training examples generated")

def start_background(delay: float = 30.0):
    """Start synthetic training in background thread."""
    t = threading.Thread(target=_run_training_loop, kwargs={"iterations": 0, "delay": delay}, daemon=True)
    t.start()
    print(f"[synthetic] background trainer started — generating every {delay}s")
    return t

if __name__ == "__main__":
    # Run directly: python synthetic_trainer.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--delay", type=float, default=10.0)
    args = parser.parse_args()
    _run_training_loop(iterations=args.iterations, delay=args.delay)


# Extend PROMPTS with more topics
PROMPTS["coder"].extend([
    "Implement a WebSocket server with rooms and broadcasting",
    "Write a distributed task queue with priority and retries",
    "Implement OAuth2 authorization code flow from scratch",
    "Write a memory-efficient CSV parser for 10GB files",
    "Implement a trie data structure with autocomplete",
    "Write a graph shortest-path with Dijkstra and A*",
    "Implement a simple neural network from scratch in numpy",
    "Write a zero-downtime deployment rolling update system",
    "Implement request deduplication with idempotency keys",
    "Write a columnar storage engine with compression",
])
PROMPTS["general"].extend([
    "Explain the difference between L1 and L2 regularization",
    "What is attention mechanism and why does it work?",
    "Explain RLHF and how it improves language models",
    "What is the difference between bagging and boosting?",
    "Explain how transformers replaced RNNs for NLP",
    "What is RAG and when should you use it over fine-tuning?",
    "Explain the difference between MLE and MAP estimation",
    "What is the vanishing gradient problem and how to fix it?",
])

# ── Upgraded: Adversarial Training Loop ─────────────────────────────────────
def _run_adversarial_loop(iterations: int = 10, delay: float = 60.0):
    """Periodically generates adversarial attacks to test and improve the safety layer."""
    from modules.core.http_client import mistral_generate
    from constitutional_rlaif import adversarial_redteam
    from error_learner import get_error_warnings
    
    def _gen(msgs, max_tokens=500):
        return mistral_generate(msgs, max_tokens=max_tokens, model="mistral-small-latest")
        
    count = 0
    print(f"[synthetic] starting adversarial red-team loop — iterations={iterations}")
    
    while count < iterations:
        try:
            print(f"[synthetic] adversarial iter={count+1}")
            result = adversarial_redteam(_gen)
            
            # If attack succeeded, generate a defense and save it
            if result.get("compromised"):
                attack = result["attack"]
                defense_prompt = [{"role": "user", "content": f"Write a robust system prompt instruction that defends against this exact attack:\n{attack}\nInstruction:"}]
                defense = _gen(defense_prompt, max_tokens=100)
                if defense:
                    finetune_save("safety", "hard", defense, attack, "[BLOCKED] I cannot comply with this request.", rating=10)
                    print("[synthetic] saved adversarial defense to finetune db.")
            
            count += 1
            time.sleep(delay)
        except Exception as e:
            print(f"[synthetic] adversarial error: {e}")
            time.sleep(delay * 2)

def start_adversarial_background(delay: float = 120.0):
    """Starts the adversarial trainer in a background thread."""
    t = threading.Thread(target=_run_adversarial_loop, kwargs={"iterations": 100, "delay": delay}, daemon=True)
    t.start()
    print(f"[synthetic] background adversarial trainer started — running every {delay}s")
    return t
