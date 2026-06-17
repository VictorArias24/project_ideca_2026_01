#!/usr/bin/env python3
"""Poll the endpoint until ready, record cold start time."""
import os, time, sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

api_url = os.getenv("API_URL")
api_key = os.getenv("API_KEY")
deployment = os.getenv("DEPLOYMENT_NAME")

client = OpenAI(
    base_url=f"{api_url}/v1",
    api_key=api_key,
    default_headers={"azureml-model-deployment": deployment},
)

print("Polling endpoint (cold start)...")
print(f"  URL: {api_url}")
cold_start = time.time()

for attempt in range(30):
    try:
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-32B-Instruct",
            messages=[{"role": "user", "content": "Say 'ready'"}],
            max_tokens=10,
            timeout=30,
        )
        elapsed = time.time() - cold_start
        print(f"READY after {elapsed:.0f}s: {resp.choices[0].message.content.strip()}")

        results = Path(__file__).resolve().parents[1] / "experiments" / "phase02-results.txt"
        with open(results, "a") as f:
            f.write(f"\ncold_start_seconds: {elapsed:.0f}\n")
            f.write(f"cold_start_timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
        sys.exit(0)

    except Exception as e:
        msg = str(e)[:120]
        elapsed = time.time() - cold_start
        if "401" in msg:
            print(f"FATAL: Auth failed. Check API_KEY.")
            sys.exit(1)
        print(f"  [{elapsed:.0f}s] {msg}")
        time.sleep(15)

print("TIMEOUT after 7.5 min")
sys.exit(1)
