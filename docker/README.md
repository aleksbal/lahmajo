# Lahmajo Docker Development Environment

This directory contains Docker configuration for running the Lahmajo RAG system with Elasticsearch.

## Quick Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services

### Elasticsearch
- **Container**: `lahmajo-elasticsearch`
- **Ports**: `9200:9200`, `9300:9300`
- **Data**: Persistent volume `es_data`
- **Health Check**: Automatic cluster health monitoring

### Lahmajo App
- **Container**: `lahmajo-app`
- **Ports**: `8000:8080` (host:container)
- **Dependencies**: Waits for Elasticsearch to be healthy
- **Volumes**: 
  - `./static:/app/static` - Web UI files
  - `./uploads:/app/uploads` - File upload directory

## Configuration

### Environment Variables
The app is pre-configured for Elasticsearch:

```yaml
VECTOR_INDEX_PROVIDER=elasticsearch
BM25_PROVIDER=elasticsearch
ELASTICSEARCH_URL=http://elasticsearch:9200
ELASTICSEARCH_INDEX=lahmajo_vectors
ELASTICSEARCH_USE_NATIVE_HYBRID=true
```

### LLM Configuration
Default configuration uses local Ollama:
```yaml
LLM_PROVIDER=ollama_local
EMBEDDING_PROVIDER=ollama_local
LLM_BASE_URL=http://host.docker.internal:11434
EMBEDDING_BASE_URL=http://host.docker.internal:11434
```

## Development Workflow

### 1. Start Services
```bash
docker-compose up -d
```

### 2. Verify Elasticsearch
```bash
curl http://localhost:9200/_cluster/health
```

### 3. Access Web UI
Open http://localhost:8000 in your browser

### 4. Test Ingestion
```bash
# Create test file
echo "Test document content" > test.txt

# Ingest via API
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test.txt"
```

### 5. Check Elasticsearch Index
```bash
curl http://localhost:9200/lahmajo_vectors/_search?pretty
```

## Customization

### Different LLM Provider
Edit `docker-compose.yml` environment variables:

```yaml
# For OpenAI
- LLM_PROVIDER=openai
- EMBEDDING_PROVIDER=openai
- OPENAI_API_KEY=your-api-key-here

# For Ollama Cloud
- LLM_PROVIDER=ollama_cloud
- EMBEDDING_PROVIDER=ollama_cloud
- LLM_BASE_URL=https://your-ollama-cloud.com
```

### Resource Limits
Adjust Elasticsearch memory:

```yaml
environment:
  - "ES_JAVA_OPTS=-Xms1g -Xmx1g"  # Increase to 1GB
```

### External Elasticsearch
To use external Elasticsearch, comment out the Elasticsearch service and update:

```yaml
environment:
  - ELASTICSEARCH_URL=http://your-external-es:9200
```

## Troubleshooting

### Elasticsearch Issues
```bash
# View logs
docker-compose logs elasticsearch

# Restart Elasticsearch
docker-compose restart elasticsearch

# Reset data (WARNING: deletes all data)
docker-compose down -v
```

### App Issues
```bash
# View logs
docker-compose logs lahmajo

# Rebuild app
docker-compose build --no-cache lahmajo

# Restart app
docker-compose restart lahmajo
```

### Network Issues
If Ollama connection fails, ensure Ollama is running and accessible:

```bash
# Test Ollama from host
curl http://localhost:11434/api/tags

# Test from container
docker-compose exec lahmajo curl http://host.docker.internal:11434/api/tags
```

## Production Considerations

For production deployment:

1. **Security**: Enable Elasticsearch security
2. **Resources**: Adjust memory and CPU limits
3. **Persistence**: Use named volumes for data
4. **Networking**: Use overlay networks for multi-host
5. **Monitoring**: Add health checks and monitoring

## Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```