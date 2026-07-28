// =============================================================================
// Azure Container Apps Module
// =============================================================================

param envName string
param appName string
param location string
param acrLoginServer string
param acrName string
param imageTag string
param env string

@secure()
param anthropicApiKey string
@secure()
param openaiApiKey string
param llmProvider string
param llmModel string

// Container App sizing per environment
var cpuCores = env == 'prod' ? '4' : '2'
var memory = env == 'prod' ? '8Gi' : '4Gi'
var minReplicas = env == 'prod' ? 1 : 0
var maxReplicas = env == 'prod' ? 5 : 3

// Log Analytics Workspace (required by Container Apps Environment)
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'law-inversionai-${env}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Container Apps Environment
resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Get ACR credentials
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
  scope: resourceGroup('rg-inversionai-shared')
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto'
      }
      registries: [
        {
          server: acrLoginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        {
          name: 'acr-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'anthropic-api-key'
          value: anthropicApiKey
        }
        {
          name: 'openai-api-key'
          value: openaiApiKey
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'inversionai'
          image: '${acrLoginServer}/inversionai:${imageTag}'
          resources: {
            cpu: json(cpuCores)
            memory: memory
          }
          env: [
            { name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }
            { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
            { name: 'LLM_PROVIDER', value: llmProvider }
            { name: 'LLM_MODEL', value: llmModel }
            { name: 'TOMOFAST_BINARY', value: '/app/Tomofast-x/tomofastx' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/_stcore/health'
                port: 8501
              }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

output appUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output appName string = containerApp.name
