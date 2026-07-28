// =============================================================================
// Azure Key Vault Module
// =============================================================================

param name string
param location string

@secure()
param anthropicApiKey string
@secure()
param openaiApiKey string

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableSoftDelete: true
    softDeleteRetentionInDays: 90
    enableRbacAuthorization: false
    accessPolicies: []
  }
}

resource secretAnthropic 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'anthropic-api-key'
  properties: {
    value: anthropicApiKey
  }
}

resource secretOpenai 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'openai-api-key'
  properties: {
    value: openaiApiKey
  }
}

output vaultUri string = keyVault.properties.vaultUri
output id string = keyVault.id
