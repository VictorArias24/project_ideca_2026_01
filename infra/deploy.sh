#!/bin/bash
set -euo pipefail

# Deploy the Bogota Building Use Classification FastAPI app to Azure Container Apps.
#
# Setup:
#   1. Copy this file's companion template and fill in your values:
#        cp infra/.env.deploy.example infra/.env.deploy
#   2. Run:
#        az login
#        bash infra/deploy.sh
#
# Security notes:
#   - Never commit infra/.env.deploy to git.
#   - The API_KEY is stored as an ACA secret (encrypted at rest), not as a plain env var.
#   - The deployed app is restricted to ALLOWED_IP.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.deploy"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: $ENV_FILE not found."
    echo "Copy infra/.env.deploy.example to infra/.env.deploy and fill in your values."
    exit 1
fi

# shellcheck source=/dev/null
set -a
source "$ENV_FILE"
set +a

# Validate required values
for var in SUBSCRIPTION_ID RESOURCE_GROUP LOCATION ALLOWED_IP WORKSPACE_NAME ENDPOINT_NAME DEPLOYMENT_NAME API_URL API_KEY; do
    value="${!var:-}"
    if [[ -z "$value" || "$value" == __FILL__* ]]; then
        echo "ERROR: Please set $var in ${ENV_FILE} before running."
        exit 1
    fi
done

# Set active subscription
az account set --subscription "$SUBSCRIPTION_ID"

# Generate safe, unique resource names
RANDOM_SUFFIX=$(openssl rand -hex 4)
APP_NAME="bogota-classify-app-${RANDOM_SUFFIX}"
ACR_NAME="bogotaacr${RANDOM_SUFFIX}"
STORAGE_NAME="bogotadata${RANDOM_SUFFIX}"
ACA_ENV_NAME="bogota-aca-env-${RANDOM_SUFFIX}"
VNET_NAME="bogota-vnet-${RANDOM_SUFFIX}"
IDENTITY_NAME="bogota-app-identity-${RANDOM_SUFFIX}"

WORKSPACE_SCOPE="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.MachineLearningServices/workspaces/${WORKSPACE_NAME}"

echo "=== Creating resources in ${RESOURCE_GROUP} (${LOCATION}) ==="
echo "App name:        ${APP_NAME}"
echo "ACR name:        ${ACR_NAME}"
echo "Storage name:    ${STORAGE_NAME}"
echo "ACA env name:    ${ACA_ENV_NAME}"
echo "VNet name:       ${VNET_NAME}"
echo "Identity name:   ${IDENTITY_NAME}"
echo "Allowed IP:      ${ALLOWED_IP}"
echo

# 1. Create Azure Container Registry
echo "Creating Azure Container Registry..."
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --location "$LOCATION" \
    --sku Basic \
    --admin-enabled false \
    --output none

# 2. Build and push Docker image
echo "Building and pushing Docker image to ACR..."
az acr build \
    --registry "$ACR_NAME" \
    --image bogota-classify-app:latest \
    --file Dockerfile \
    --output none \
    "$(dirname "$0")/.."

# 3. Create Virtual Network with ACA subnet
echo "Creating Virtual Network..."
az network vnet create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VNET_NAME" \
    --location "$LOCATION" \
    --address-prefix 10.0.0.0/16 \
    --subnet-name aca-subnet \
    --subnet-prefix 10.0.0.0/23 \
    --output none

ACA_SUBNET_ID="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Network/virtualNetworks/${VNET_NAME}/subnets/aca-subnet"

# 4. Create Azure Storage account and file share for SQLite
echo "Creating Azure Storage account and file share..."
az storage account create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$STORAGE_NAME" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none

STORAGE_KEY=$(az storage account keys list \
    --resource-group "$RESOURCE_GROUP" \
    --account-name "$STORAGE_NAME" \
    --query '[0].value' \
    --output tsv)

az storage share create \
    --account-name "$STORAGE_NAME" \
    --name sqlite-data \
    --account-key "$STORAGE_KEY" \
    --output none

# 5. Create user-assigned managed identity
echo "Creating user-assigned managed identity..."
az identity create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$IDENTITY_NAME" \
    --location "$LOCATION" \
    --output none

IDENTITY_CLIENT_ID=$(az identity show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$IDENTITY_NAME" \
    --query clientId \
    --output tsv)

IDENTITY_RESOURCE_ID=$(az identity show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$IDENTITY_NAME" \
    --query id \
    --output tsv)

# Wait for identity propagation
sleep 30

# 6. Grant identity access to Azure ML workspace
echo "Granting managed identity access to Azure ML workspace..."
az role assignment create \
    --assignee "$IDENTITY_CLIENT_ID" \
    --role "Azure Machine Learning Data Scientist" \
    --scope "$WORKSPACE_SCOPE" \
    --output none

# 7. Grant identity pull access to ACR
echo "Granting managed identity ACR pull access..."
ACR_SCOPE=$(az acr show --name "$ACR_NAME" --query id --output tsv)
az role assignment create \
    --assignee "$IDENTITY_CLIENT_ID" \
    --role AcrPull \
    --scope "$ACR_SCOPE" \
    --output none

# 8. Create Container Apps environment in the VNet
echo "Creating Container Apps environment..."
az containerapp env create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACA_ENV_NAME" \
    --location "$LOCATION" \
    --infrastructure-subnet-resource-id "$ACA_SUBNET_ID" \
    --output none

# 9. Add Azure Files storage to the environment
echo "Adding Azure Files storage volume..."
az containerapp env storage set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACA_ENV_NAME" \
    --storage-name sqlite \
    --azure-file-account-name "$STORAGE_NAME" \
    --azure-file-share-name sqlite-data \
    --azure-file-account-key "$STORAGE_KEY" \
    --access-mode ReadWrite \
    --output none

# 10. Create the Container App
echo "Creating Container App..."
ENV_VARS=(
    "DATABASE_URL=sqlite+aiosqlite:////data/classifications.db"
    "API_URL=${API_URL}"
    "API_KEY=secretref:api-key"
    "ENDPOINT_NAME=${ENDPOINT_NAME}"
    "DEPLOYMENT_NAME=${DEPLOYMENT_NAME}"
    "SUBSCRIPTION_ID=${SUBSCRIPTION_ID}"
    "RESOURCE_GROUP=${RESOURCE_GROUP}"
    "WORKSPACE_NAME=${WORKSPACE_NAME}"
    "LOCATION=${LOCATION}"
    "AZURE_CLIENT_ID=${IDENTITY_CLIENT_ID}"
    "FORCE_MOCK=0"
)

if [[ -n "${ADMIN_API_KEY:-}" && "$ADMIN_API_KEY" != __FILL__* ]]; then
    ENV_VARS+=("ADMIN_API_KEY=secretref:admin-api-key")
fi

az containerapp create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --environment "$ACA_ENV_NAME" \
    --image "${ACR_NAME}.azurecr.io/bogota-classify-app:latest" \
    --target-port 7860 \
    --ingress external \
    --min-replicas 1 \
    --max-replicas 1 \
    --user-assigned "$IDENTITY_RESOURCE_ID" \
    --env-vars "${ENV_VARS[@]}" \
    --secrets "api-key=${API_KEY}" \
    --cpu 1 \
    --memory 2Gi \
    --output none

if [[ -n "${ADMIN_API_KEY:-}" && "$ADMIN_API_KEY" != __FILL__* ]]; then
    az containerapp secret set \
        --resource-group "$RESOURCE_GROUP" \
        --name "$APP_NAME" \
        --secrets "admin-api-key=${ADMIN_API_KEY}" \
        --output none
fi

# 11. Mount SQLite volume
echo "Mounting SQLite volume..."
az containerapp update \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --storage-mount "/data=sqlite" \
    --output none

# 12. Restrict public access to allowed IP
echo "Setting IP access restriction..."
az containerapp ingress access-restriction set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --rule-name allow-client \
    --ip-address "$ALLOWED_IP" \
    --action Allow \
    --output none

# 13. Show the public URL
echo
echo "=== Deployment complete ==="
APP_FQDN=$(az containerapp show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --query properties.configuration.ingress.fqdn \
    --output tsv)

echo "Public URL: https://${APP_FQDN}"
echo "Allowed IP: ${ALLOWED_IP}"
echo
if [[ -n "${ADMIN_API_KEY:-}" && "$ADMIN_API_KEY" != __FILL__* ]]; then
    echo "Admin API key is configured and required for /admin/*, /batch/upload, and /jobs endpoints."
fi
echo
echo "To verify:"
echo "  curl -s https://${APP_FQDN}/health"
echo
echo "To clean up everything created by this script:"
echo "  az group delete --name ${RESOURCE_GROUP} --yes --no-wait"
