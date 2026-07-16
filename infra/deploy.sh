#!/usr/bin/env bash
# =============================================================================
# InversionAI - Azure Deployment Script
# =============================================================================
# Usage:
#   ./deploy.sh dev     # Deploy dev environment
#   ./deploy.sh staging # Deploy staging environment
#   ./deploy.sh prod    # Deploy production environment
# =============================================================================

set -euo pipefail

ENV="${1:-dev}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$SCRIPT_DIR"
PROJECT_DIR="$(dirname "$INFRA_DIR")"

# Validate environment
if [[ ! "$ENV" =~ ^(dev|staging|prod)$ ]]; then
    echo "❌ Invalid environment: $ENV (must be dev, staging, or prod)"
    exit 1
fi

echo "🚀 Deploying InversionAI to $ENV environment..."
echo ""

# --- Step 1: Build and push Docker image ---
echo "📦 Building Docker image (tag: ${ENV}-latest)..."
cd "$PROJECT_DIR"
az acr build \
    --registry inversionaicr \
    --image "inversionai:${ENV}-latest" \
    --platform linux/amd64 \
    . --no-logs

echo "✅ Image pushed: inversionaicr.azurecr.io/inversionai:${ENV}-latest"
echo ""

# --- Step 2: Deploy infrastructure ---
echo "🏗️  Deploying infrastructure..."

# Check if secrets are in parameter file (they shouldn't be empty)
PARAMS_FILE="$INFRA_DIR/parameters.${ENV}.json"
if grep -q '"anthropicApiKey": { "value": "" }' "$PARAMS_FILE"; then
    echo "⚠️  API keys are empty in $PARAMS_FILE"
    echo "   Pass them via CLI: --parameters anthropicApiKey=<key> openaiApiKey=<key>"
    echo ""
    read -p "Enter ANTHROPIC_API_KEY: " -s ANTHROPIC_KEY
    echo ""
    read -p "Enter OPENAI_API_KEY: " -s OPENAI_KEY
    echo ""

    az deployment sub create \
        --location australiaeast \
        --template-file "$INFRA_DIR/main.bicep" \
        --parameters "@$PARAMS_FILE" \
        --parameters "anthropicApiKey=$ANTHROPIC_KEY" "openaiApiKey=$OPENAI_KEY" \
        --name "inversionai-${ENV}-$(date +%Y%m%d%H%M%S)"
else
    az deployment sub create \
        --location australiaeast \
        --template-file "$INFRA_DIR/main.bicep" \
        --parameters "@$PARAMS_FILE" \
        --name "inversionai-${ENV}-$(date +%Y%m%d%H%M%S)"
fi

echo ""
echo "✅ Deployment complete!"
echo ""

# --- Step 3: Show outputs ---
APP_URL=$(az containerapp show \
    --name "ca-inversionai-${ENV}" \
    --resource-group "rg-inversionai-${ENV}" \
    --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🌐 App URL:     https://$APP_URL"
echo "  📦 Image:       inversionaicr.azurecr.io/inversionai:${ENV}-latest"
echo "  🔑 Key Vault:   kv-inversionai-${ENV}"
echo "  💾 Storage:     stinversionai${ENV}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
