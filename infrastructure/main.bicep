// LLM Eval CI/CD — Azure Container Apps deployment
// Deploy with: az deployment group create --resource-group <rg> --template-file main.bicep --parameters @main.parameters.json

@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment name: dev, staging, prod')
param environmentName string = 'prod'

@description('Container image tag to deploy')
param imageTag string = 'latest'

@description('Azure Container Registry name')
param acrName string

@description('Docker image name')
param imageName string = 'llm-eval-cicd'

@description('PostgreSQL admin password — inject at deploy time, never hard-code')
@secure()
param postgresAdminPassword string

@description('OpenAI API key — inject at deploy time, never hard-code')
@secure()
param openaiApiKey string = ''

@description('Application Insights connection string — injected at deploy time')
@secure()
param appInsightsConnectionString string = ''

var prefix = 'llmeval-${environmentName}'
var tags = { project: 'llm-eval-cicd', environment: environmentName }

// ─── Log Analytics Workspace ──────────────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${prefix}-logs'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ─── Application Insights ─────────────────────────────────────────────────────
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-insights'
  location: location
  kind: 'web'
  tags: tags
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// ─── Container Apps Environment ───────────────────────────────────────────────
resource containerAppsEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${prefix}-env'
  location: location
  tags: tags
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

// ─── PostgreSQL Flexible Server ───────────────────────────────────────────────
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2023-06-01-preview' = {
  name: '${prefix}-pg'
  location: location
  tags: tags
  sku: { name: 'Standard_B1ms', tier: 'Burstable' }
  properties: {
    administratorLogin: 'evaladmin'
    administratorLoginPassword: postgresAdminPassword
    version: '16'
    storage: { storageSizeGB: 32 }
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
  }
}

resource evalDatabase 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-06-01-preview' = {
  parent: postgres
  name: 'llmeval'
}

// ─── Storage Account (eval results & artifacts) ───────────────────────────────
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: replace('${prefix}store', '-', '')
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    supportsHttpsTrafficOnly: true
  }
}

resource evalContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/eval-results'
  properties: { publicAccess: 'None' }
}

// ─── Container App — Backend API ──────────────────────────────────────────────
resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${prefix}-api'
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: containerAppsEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
      }
      registries: [
        {
          server: '${acrName}.azurecr.io'
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'llm-eval-api'
          image: '${acrName}.azurecr.io/${imageName}:${imageTag}'
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'DATABASE_URL', value: 'postgresql://evaladmin:${postgresAdminPassword}@${postgres.properties.fullyQualifiedDomainName}:5432/llmeval?sslmode=require' }
            { name: 'STORAGE_BACKEND', value: 'azure' }
            { name: 'AZURE_STORAGE_ACCOUNT_NAME', value: storageAccount.name }
            { name: 'AZURE_STORAGE_CONTAINER', value: 'eval-results' }
            { name: 'LLM_PROVIDER', value: 'mock' }
            { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights-conn' }
            { name: 'LOG_LEVEL', value: 'INFO' }
          ]
        }
      ]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

// ─── Outputs ─────────────────────────────────────────────────────────────────
output backendUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output postgresHost string = postgres.properties.fullyQualifiedDomainName
output storageAccountName string = storageAccount.name
output appInsightsName string = appInsights.name
