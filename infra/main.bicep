// =============================================================================
// InversionAI - Azure Infrastructure (Main)
// =============================================================================
// Deploy with:
//   az deployment sub create --location australiaeast --template-file infra/main.bicep --parameters env=dev
//   az deployment sub create --location australiaeast --template-file infra/main.bicep --parameters env=staging
//   az deployment sub create --location australiaeast --template-file infra/main.bicep --parameters env=prod

targetScope = 'subscription'

@allowed(['dev', 'staging', 'prod'])
param env string

param location string = 'australiaeast'
param acrName string = 'inversionaicr'
param imageTag string = '${env}-latest'

// Secrets (pass via parameter file or --parameters)
@secure()
param anthropicApiKey string
@secure()
param openaiApiKey string

param llmProvider string = 'anthropic'
param llmModel string = 'claude-sonnet-4-20250514'

// ---------------------------------------------------------------------------
// Resource Groups
// ---------------------------------------------------------------------------
resource rgShared 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-inversionai-shared'
  location: location
  tags: {
    project: 'inversionai'
    environment: 'shared'
  }
}

resource rgEnv 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-inversionai-${env}'
  location: location
  tags: {
    project: 'inversionai'
    environment: env
  }
}

// ---------------------------------------------------------------------------
// Shared Resources (ACR)
// ---------------------------------------------------------------------------
module acr 'modules/acr.bicep' = {
  name: 'acr-deployment'
  scope: rgShared
  params: {
    name: acrName
    location: location
  }
}

// ---------------------------------------------------------------------------
// Environment Resources
// ---------------------------------------------------------------------------
module keyVault 'modules/keyvault.bicep' = {
  name: 'keyvault-deployment-${env}'
  scope: rgEnv
  params: {
    name: 'kv-inversionai-${env}'
    location: location
    anthropicApiKey: anthropicApiKey
    openaiApiKey: openaiApiKey
  }
}

module storage 'modules/storage.bicep' = {
  name: 'storage-deployment-${env}'
  scope: rgEnv
  params: {
    name: 'stinversionai${env}'
    location: location
  }
}

module containerApp 'modules/containerapp.bicep' = {
  name: 'containerapp-deployment-${env}'
  scope: rgEnv
  params: {
    envName: 'cae-inversionai-${env}'
    appName: 'ca-inversionai-${env}'
    location: location
    acrLoginServer: acr.outputs.loginServer
    acrName: acrName
    imageTag: imageTag
    anthropicApiKey: anthropicApiKey
    openaiApiKey: openaiApiKey
    llmProvider: llmProvider
    llmModel: llmModel
    env: env
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------
output appUrl string = containerApp.outputs.appUrl
output acrLoginServer string = acr.outputs.loginServer
output keyVaultUri string = keyVault.outputs.vaultUri
output storageAccountName string = storage.outputs.accountName
