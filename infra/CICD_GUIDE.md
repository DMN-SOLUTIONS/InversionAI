# InversionAI - CI/CD Workflow Guide

## Git Branching Strategy

```
production  ← stable releases (manual approval to deploy)
  ↑
staging     ← pre-production testing
  ↑
main        ← development (auto-deploys to dev environment)
  ↑
feature/*   ← your working branches
```

## How It Works

| Action | Result |
|--------|--------|
| Push to `main` | Tests → Build → Deploy to **dev** |
| Push to `staging` | Tests → Build → Deploy to **staging** |
| Push to `production` | Tests → Build → **Manual approval** → Deploy to **prod** |
| Pull Request to `main` | Tests only (no deploy) |

## Initial Setup

### 1. Create a GitHub repository

```bash
cd /path/to/InversionAI
git init
git add .
git commit -m "Initial commit"
gh repo create DavionAI/InversionAI --private --source=. --push
```

### 2. Create Azure Service Principal for GitHub Actions

```bash
az ad sp create-for-rbac \
  --name "github-inversionai" \
  --role contributor \
  --scopes /subscriptions/c44aab7c-f481-4ddc-b99f-27f001ca3aac \
  --sdk-auth
```

Copy the JSON output — you'll need it for the next step.

### 3. Add GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret Name | Value |
|---|---|
| `AZURE_CREDENTIALS` | The JSON from step 2 |
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `OPENAI_API_KEY` | Your OpenAI API key |

### 4. Configure Environment Protection (for production)

Go to **Settings → Environments** and:

1. Create environment: `production`
2. Add **Required reviewers** (yourself)
3. This means production deploys require your manual approval

### 5. Create the staging and production branches

```bash
git checkout -b staging
git push -u origin staging

git checkout -b production
git push -u origin production

git checkout main
```

## Daily Workflow

### Making changes and deploying to dev:

```bash
# Create a feature branch
git checkout -b feature/my-change

# Make your changes...
# ...

# Push and create PR
git add .
git commit -m "Add new feature"
git push -u origin feature/my-change
gh pr create --base main --title "Add new feature"

# After PR is reviewed/approved, merge to main
gh pr merge --squash

# This automatically deploys to dev!
```

### Promoting to staging:

```bash
# Merge main into staging
git checkout staging
git pull
git merge main
git push

# This automatically deploys to staging!
```

### Promoting to production:

```bash
# Merge staging into production
git checkout production
git pull
git merge staging
git push

# This triggers the workflow but WAITS for your manual approval
# Go to GitHub Actions → approve the deployment
```

## Quick Commands

```bash
# Check deployment status
az containerapp show --name ca-inversionai-dev --resource-group rg-inversionai-dev --query "properties.runningStatus" -o tsv

# View logs
az containerapp logs show --name ca-inversionai-dev --resource-group rg-inversionai-dev --type console --tail 50

# Rollback to previous image
az containerapp update --name ca-inversionai-dev --resource-group rg-inversionai-dev --image inversionaicr.azurecr.io/inversionai:dev-latest
```

## Before First Staging/Prod Deploy

You need to deploy the staging and production infrastructure first:

```bash
cd infra
./deploy.sh staging
./deploy.sh prod
```

Or use Bicep directly:

```bash
az deployment sub create --location australiaeast --template-file infra/main.bicep \
  --parameters @infra/parameters.staging.json \
  --parameters anthropicApiKey=<key> openaiApiKey=<key>
```
