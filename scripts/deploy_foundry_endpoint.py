#!/usr/bin/env python3
"""
Deploy Qwen2.5-VL-32B-Instruct as an Azure ML Managed Online Endpoint
via the Hugging Face collection on Microsoft Foundry.

Reference: https://huggingface.co/docs/microsoft-azure/foundry/examples/deploy-vision-language-models

Usage:
    python scripts/deploy_foundry_endpoint.py

Requirements:
    - .env file with SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME, LOCATION
    - az login completed
    - Hub-based Foundry project
"""

import os
import sys
import time
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential

# --- Configuration ---
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME")
LOCATION = os.getenv("LOCATION", "eastus")

for var_name, var_val in [
    ("SUBSCRIPTION_ID", SUBSCRIPTION_ID),
    ("RESOURCE_GROUP", RESOURCE_GROUP),
    ("WORKSPACE_NAME", WORKSPACE_NAME),
]:
    if not var_val or var_val.startswith("<"):
        print(f"ERROR: {var_name} not set or contains placeholder in .env")
        sys.exit(1)

# Endpoint and deployment names (must be unique per region)
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME") or f"bogota-vlm-{str(uuid4())[:8]}"
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME") or f"qwen-vl-{str(uuid4())[:8]}"

# Model: Qwen2.5-VL-32B-Instruct
MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"
MODEL_URI = (
    f"azureml://registries/HuggingFace/models/"
    f"{MODEL_NAME.replace('/', '-').replace('_', '-').lower()}/labels/latest"
)

# Instance type: A100-80GB
INSTANCE_TYPE = "Standard_NC24ads_A100_v4"

# --- Authenticate ---
print("=" * 60)
print("DEPLOYING VLM MANAGED ONLINE ENDPOINT")
print("=" * 60)
print(f"Endpoint:   {ENDPOINT_NAME}")
print(f"Deployment: {DEPLOYMENT_NAME}")
print(f"Model:      {MODEL_NAME}")
print(f"Instance:   {INSTANCE_TYPE}")
print(f"Region:     {LOCATION}")
print()

print("Authenticating to Azure...")
credential = DefaultAzureCredential()

client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

ws = client.workspaces.get(WORKSPACE_NAME)
print(f"Workspace: {ws.name} ({ws.location})")
print()

# --- Create Endpoint ---
print(f"Creating endpoint '{ENDPOINT_NAME}'...")
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Bogota land-use classification VLM endpoint (Phase 02)",
    auth_mode="key",
)

try:
    client.begin_create_or_update(endpoint).wait()
    print(f"  OK: endpoint created")
except Exception as e:
    print(f"  FAIL: {e}")
    print("\nCheck:")
    print("  1. Quota for Standard_NC24ads_A100_v4 in your region")
    print("  2. Resource group and workspace names")
    print("  3. Azure CLI login (az login)")
    sys.exit(1)

# --- Create Deployment ---
print(f"\nCreating deployment '{DEPLOYMENT_NAME}'...")
print(f"  Model URI: {MODEL_URI}")
print(f"  Instance:  {INSTANCE_TYPE} (1 instance)")
print(f"  Timeout:   120s")
print()

deployment_start = time.time()

deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=MODEL_URI,
    instance_type=INSTANCE_TYPE,
    instance_count=1,
    request_settings={
        "max_concurrent_requests_per_instance": 1,
        "request_timeout_ms": 120000,
    },
)

try:
    client.online_deployments.begin_create_or_update(deployment).wait()
    deployment_elapsed = time.time() - deployment_start
    print(f"  OK: deployment ready in {deployment_elapsed/60:.1f} minutes")
except Exception as e:
    print(f"  FAIL: {e}")
    print("\nCheck:")
    print("  1. GPU quota: Azure Portal -> Machine Learning -> Quota")
    print("  2. Model catalog: https://ml.azure.com -> search 'Qwen'")
    print("  3. Project is Hub-based (required for HuggingFace collection)")
    sys.exit(1)

# --- Set Traffic ---
print(f"\nSetting 100% traffic to '{DEPLOYMENT_NAME}'...")
endpoint.traffic = {DEPLOYMENT_NAME: 100}
client.begin_create_or_update(endpoint).wait()
print("  OK: traffic routed")

# --- Get Keys and URL ---
ep = client.online_endpoints.get(ENDPOINT_NAME)
scoring_uri = ep.scoring_uri
api_url = f"https://{ENDPOINT_NAME}.{LOCATION}.inference.ml.azure.com/v1"

keys = client.online_endpoints.get_keys(ENDPOINT_NAME)
api_key = keys.primary_key

# --- Summary ---
print()
print("=" * 60)
print("DEPLOYMENT COMPLETE")
print("=" * 60)
print(f"Endpoint:    {ENDPOINT_NAME}")
print(f"Deployment:  {DEPLOYMENT_NAME}")
print(f"Model:       {MODEL_NAME}")
print(f"API URL:     {api_url}")
print(f"Scoring URI: {scoring_uri}")
print(f"API Key:     {api_key[:8]}...{api_key[-4:]}")
print()

# --- Update .env ---
env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
env_lines = [
    f"\n# Phase 02 — Managed Online Endpoint ({time.strftime('%Y-%m-%d %H:%M')})",
    f"ENDPOINT_NAME={ENDPOINT_NAME}",
    f"DEPLOYMENT_NAME={DEPLOYMENT_NAME}",
    f"API_KEY={api_key}",
    f"API_URL={api_url}",
    f"SCORING_URI={scoring_uri}",
]

print("Updating .env ...")
with open(env_file, "a") as f:
    f.write("\n".join(env_lines) + "\n")
print(f"  OK: credentials written to .env")

# --- Record Deployment Time ---
os.makedirs("experiments", exist_ok=True)
with open("experiments/phase02-results.txt", "a") as f:
    f.write(f"deployment_time_minutes: {deployment_elapsed/60:.1f}\n")
    f.write(f"endpoint_name: {ENDPOINT_NAME}\n")
    f.write(f"deployment_name: {DEPLOYMENT_NAME}\n")
    f.write(f"model: {MODEL_NAME}\n")
    f.write(f"instance_type: {INSTANCE_TYPE}\n")
    f.write(f"location: {LOCATION}\n")
    f.write(f"timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
print("  OK: results recorded to experiments/phase02-results.txt")

print("\nDone. Run scripts/test_endpoint_single.py to verify.")
