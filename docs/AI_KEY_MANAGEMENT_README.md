# AI Key Management System

A comprehensive API key management system for AI providers (OpenAI, Anthropic, Google) with health monitoring, automatic retry logic, and real-time key pool updates.

## 🚀 Features

### 1. Initial Key Health Checks
- Validates all API keys before adding them to the pool
- Removes invalid and quota-exceeded keys immediately
- Fast format validation for quick filtering

### 2. Smart Key Pool Management
- Organizes keys by provider (OpenAI, Anthropic, Google)
- Real-time status tracking for each key
- Automatic removal of failed keys
- Success rate monitoring

### 3. Universal Retry Logic
- Configurable retry attempts (default: 3, via `MAX_RETRIES` env var)
- Intelligent error classification per provider
- Automatic key rotation on failures
- Exponential backoff between retries

### 4. Real-time Pool Updates
- Keys are immediately removed when they fail with permanent errors
- Background health monitoring (configurable interval)
- Firebase integration for persistent tracking
- Dashboard-ready analytics

### 5. Provider-Specific Error Handling

#### OpenAI
- `401` → Invalid Key (remove from pool)
- `429` → Quota Exceeded (remove from pool)
- Other errors → Keep in pool, retry

#### Anthropic
- `401` → Invalid Key (remove from pool)
- `429` → Rate Limited (keep in pool, temporary)
- `403` → Quota Exceeded (remove from pool)
- Other errors → Keep in pool, retry

#### Google
- `403` → Invalid Key (remove from pool)
- `429` → Rate Limited (keep in pool, temporary)
- No quota exceeded handling for Google
- Other errors → Keep in pool, retry

## 📦 Installation

1. Copy the `core/` directory to your project
2. Install required dependencies:

```bash
pip install requests firebase-admin
# Plus your AI SDK dependencies:
pip install openai anthropic google-generativeai
```

## 🔧 Configuration

Set environment variables:

```bash
# Required: API Keys (comma-separated)
export OPENAI_API_KEYS="sk-key1,sk-key2,sk-key3"
export ANTHROPIC_API_KEYS="sk-ant-key1,sk-ant-key2"
export GOOGLE_API_KEYS="AIzaSy-key1,AIzaSy-key2"

# Optional: Configuration
export MAX_RETRIES=3                          # Default: 3
export HEALTH_CHECK_INTERVAL_MINUTES=10       # Default: 10

# Optional: Firebase (for persistence and analytics)
export FIREBASE_URL="your-firebase-url"
export FIREBASE_CREDENTIALS_JSON="path/to/credentials.json"
```

## 🚀 Quick Start

### Basic Usage

```python
from core.ai_key_manager import (
    initialize_openai_keys,
    initialize_anthropic_keys,
    execute_openai_call,
    execute_anthropic_call
)

# 1. Initialize keys with health checks
openai_keys = ["sk-key1", "sk-key2", "sk-key3"]
result = initialize_openai_keys(openai_keys)
print(f"✅ {result['healthy_keys_added']} OpenAI keys ready")

# 2. Define your API call function
def my_openai_call(message: str, model: str, api_key: str):
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}]
    )
    return {"content": response.choices[0].message.content}

# 3. Execute with automatic retry and key management
result = execute_openai_call(my_openai_call, 
                           message="Hello!", 
                           model="gpt-4o")

if result.success:
    print(f"✅ Success: {result.response['content']}")
    print(f"📊 Attempts: {result.attempts_made}")
    print(f"🔑 Keys tried: {result.keys_tried}")
else:
    print(f"❌ Failed: {result.error_message}")
```

### Advanced Usage

```python
from core.ai_key_manager import get_ai_key_manager

# Get the global manager for advanced operations
manager = get_ai_key_manager()

# Check system status
status = manager.get_comprehensive_status()
print(f"Total requests: {status['global_stats']['total_requests']}")
print(f"Keys removed: {status['global_stats']['keys_removed']}")

# Force health check
health_result = manager.force_health_check()

# Get provider analytics
analytics = manager.get_provider_analytics('openai')
print(f"OpenAI health: {analytics['summary']['health_percentage']}%")

# Add new keys dynamically
new_keys = ["sk-new-key1", "sk-new-key2"]
add_result = manager.add_new_keys('openai', new_keys)
print(f"Added {add_result['healthy_keys_added']} new keys")

# Update configuration
manager.update_configuration(max_retries=5, health_check_interval=15)
```

## 📊 Monitoring & Analytics

### System Health Endpoint

```python
from flask import Flask, jsonify
from core.ai_key_manager import get_ai_key_manager

app = Flask(__name__)

@app.route('/health/keys')
def key_health():
    manager = get_ai_key_manager()
    status = manager.get_comprehensive_status()
    return jsonify(status)

@app.route('/analytics/<provider>')
def provider_analytics(provider):
    manager = get_ai_key_manager()
    analytics = manager.get_provider_analytics(provider)
    return jsonify(analytics)
```

### Dashboard Data Structure

The system provides comprehensive data for building dashboards:

```json
{
  "system_info": {
    "max_retries": 3,
    "initialized_providers": ["openai", "anthropic"],
    "firebase_available": true,
    "health_check_interval_minutes": 10
  },
  "key_pools": {
    "providers": {
      "openai": {
        "total_keys": 5,
        "healthy_keys": 3,
        "removed_keys": 2,
        "status_breakdown": {
          "healthy": 3,
          "invalid_key": 1,
          "quota_exceeded": 1
        },
        "keys": [
          {
            "masked_key": "sk-1234...cdef",
            "status": "healthy",
            "success_rate": 95.5,
            "response_time": 1.2,
            "error_count": 1,
            "success_count": 21
          }
        ]
      }
    }
  },
  "global_stats": {
    "total_requests": 150,
    "total_failures": 12,
    "keys_removed": 3
  }
}
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_key_system.py
```

This will:
1. Test key initialization with mock keys
2. Simulate API calls with different error scenarios
3. Demonstrate retry logic and key rotation
4. Show system analytics and status

## 🔄 Integration with Existing Code

### Replace Existing API Calls

**Before (manual key management):**
```python
import openai

def call_openai(message):
    for key in api_keys:
        try:
            client = openai.OpenAI(api_key=key)
            response = client.chat.completions.create(...)
            return response
        except Exception as e:
            if "invalid" in str(e):
                # Manual key removal logic
                continue
            # Manual retry logic
```

**After (automated management):**
```python
from core.ai_key_manager import execute_openai_call

def my_openai_call(message, api_key):
    client = openai.OpenAI(api_key=api_key)
    return client.chat.completions.create(...)

def call_openai(message):
    result = execute_openai_call(my_openai_call, message=message)
    return result.response if result.success else None
```

### Flask/FastAPI Integration

```python
from flask import Flask, request, jsonify
from core.ai_key_manager import execute_openai_call

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    
    result = execute_openai_call(my_openai_call, message=message)
    
    if result.success:
        return jsonify({
            "response": result.response['content'],
            "metadata": {
                "attempts": result.attempts_made,
                "total_time": result.total_time
            }
        })
    else:
        return jsonify({"error": result.error_message}), 500
```

## 🔍 Error Handling

The system handles errors according to your specifications:

- **Invalid keys** → Immediately removed, not retried
- **Quota exceeded** → Immediately removed, not retried  
- **Rate limiting** → Kept in pool, retried with different key
- **Network errors** → Retried with exponential backoff
- **Server errors** → Retried with different key

Failed attempts are logged with full context for debugging.

## 🔧 Customization

### Custom Error Classification

```python
from core.retry_logic import UniversalRetrySystem

# Extend error classification for new providers
retry_system = UniversalRetrySystem(key_pool_manager)
retry_system.error_classifications['custom_provider'] = {
    401: 'invalid_key',
    402: 'quota_exceeded',
    429: 'rate_limited'
}
```

### Custom Health Checkers

```python
from core.health_checker import BaseHealthChecker, HealthResult

class CustomHealthChecker(BaseHealthChecker):
    def check_health(self, api_key: str) -> HealthResult:
        # Implement custom health check logic
        pass
    
    def classify_error(self, status_code: int, error_message: str):
        # Implement custom error classification
        pass

# Add to health manager
health_manager.add_checker('custom_provider', CustomHealthChecker())
```

## 📈 Performance Optimization

- **Fast health checks**: Uses lightweight endpoints when available
- **Concurrent initialization**: Health checks run in parallel
- **Efficient key selection**: Best keys selected based on success rate and usage
- **Background monitoring**: Health checks don't block API calls
- **Firebase caching**: Persistent storage reduces initialization time

## 🛠️ Troubleshooting

### Common Issues

1. **No healthy keys available**
   - Check API key format and validity
   - Verify network connectivity
   - Check quotas and billing status

2. **High retry rates**
   - Monitor key health dashboard
   - Check for quota limits being hit
   - Verify correct error classification

3. **Performance issues**
   - Adjust health check interval
   - Monitor Firebase connection
   - Check key pool sizes

### Debug Mode

Set debug logging to see detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_RETRIES` | 3 | Maximum retry attempts per API call |
| `HEALTH_CHECK_INTERVAL_MINUTES` | 10 | Minutes between automatic health checks |
| `OPENAI_API_KEYS` | - | Comma-separated OpenAI API keys |
| `ANTHROPIC_API_KEYS` | - | Comma-separated Anthropic API keys |
| `GOOGLE_API_KEYS` | - | Comma-separated Google API keys |
| `FIREBASE_URL` | - | Firebase Realtime Database URL |
| `FIREBASE_CREDENTIALS_JSON` | - | Path to Firebase credentials |

## 🤝 Contributing

To extend the system for new AI providers:

1. Add health checker in `core/health_checker.py`
2. Add error classification in `core/retry_logic.py`
3. Add convenience functions in `core/ai_key_manager.py`
4. Update tests and documentation

## 📄 License

This AI Key Management System is part of your proxy application and follows the same licensing terms.