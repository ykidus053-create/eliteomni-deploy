import asyncio, random, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
MISTRAL_KEY = os.environ.get("MISTRAL_API_KEY", "")

async def train_one(client, prompt, skill):
    try:
        headers = {"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type": "application/json"}

        if skill == "coder":
            gen_model = "mistral-code-agent-latest"
        elif skill in ("general", "researcher"):
            gen_model = "magistral-medium-latest"
        else:
            gen_model = "mistral-small-latest"

        # Generate — API latency naturally throttles us
        r = await client.post("https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json={"model": gen_model, "max_tokens": 4000,
                  "messages": [
                      {"role": "system", "content": "You are an expert. Give complete, runnable answers with no placeholders."},
                      {"role": "user", "content": prompt}
                  ]},
            timeout=60)

        if r.status_code == 429:
            print(f"[fast] rate limited — sleeping 60s", flush=True)
            await asyncio.sleep(60)
            return

        response = r.json()["choices"][0]["message"]["content"]
        if len(response) < 100:
            return

        # Score
        r2 = await client.post("https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json={"model": "mistral-small-latest", "max_tokens": 5,
                  "messages": [{"role": "user", "content":
                      f"Score 0.0-1.0. Reply ONLY with float.\n1.0=complete\n0.5=partial\n0.1=fake\nTask:{prompt[:150]}\nResponse:{response[:1000]}\nScore:"}]},
            timeout=15)

        if r2.status_code == 429:
            score = 0.7 if len(response) > 500 else 0.4
        else:
            match = re.search(r'0\.\d+|1\.0', r2.json()["choices"][0]["message"]["content"])
            score = float(match.group()) if match else 0.5

        if score >= 0.4:
            from finetune import finetune_save
            finetune_save(skill, "hard" if skill=="coder" else "medium",
                          "You are an expert.", prompt, response,
                          rating=int(score * 10))
            print(f"[fast] saved score={score:.1f} model={gen_model} len={len(response)}", flush=True)

    except Exception as e:
        print(f"[fast] error: {e}", flush=True)

async def run_fast(concurrency=1, rounds=999999):
    from synthetic_trainer import PROMPTS
    import httpx
    async with httpx.AsyncClient() as client:
        for i in range(rounds):
            skill = random.choice(["coder", "coder", "general", "researcher"])
            prompt = random.choice(PROMPTS[skill])
            print(f"[fast] sample={i+1} skill={skill} prompt={prompt[:50]}", flush=True)
            await train_one(client, prompt, skill)
            # No artificial delay — API response time (~15s) is the natural throttle

if __name__ == "__main__":
    asyncio.run(run_fast())
