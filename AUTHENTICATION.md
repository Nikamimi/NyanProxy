# NyanProxy Authentication System

## Overview

NyanProxy now includes a comprehensive authentication and user management system with dual authentication modes, usage tracking, and Firebase integration.

## 🔐 Authentication Modes

### 1. No Authentication (`AUTH_MODE=none`)
- No authentication required
- All requests are allowed
- No user tracking or quotas

### 2. Proxy Key Mode (`AUTH_MODE=proxy_key`)
- Single shared password for all users
- Set `PROXY_PASSWORD` environment variable
- IP-based rate limiting
- No individual user tracking

### 3. User Token Mode (`AUTH_MODE=user_token`)
- Individual UUID tokens for each user
- Complete user management with Firebase storage
- Per-user quotas and usage tracking
- IP address monitoring and limits
- Ban/unban functionality

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
# Basic configuration
AUTH_MODE=user_token
ADMIN_KEY=your_admin_key_here

# Firebase (for user_token mode)
FIREBASE_URL=https://your-project-default-rtdb.firebaseio.com/

# API Keys
OPENAI_API_KEYS=sk-key1,sk-key2
ANTHROPIC_API_KEYS=sk-ant-key1,sk-ant-key2
```

### 3. Start the Server
```bash
python app.py
```

## 🔑 Authentication Headers

### For Proxy Key Mode:
```bash
curl -H "Authorization: Bearer your_proxy_password" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello"}]}' \
     https://your-proxy.com/v1/chat/completions
```

### For User Token Mode:
```bash
curl -H "Authorization: Bearer user_token_uuid" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Hello"}]}' \
     https://your-proxy.com/v1/chat/completions
```

### Alternative Headers:
- `x-api-key: your_token`
- Query parameter: `?key=your_token`

## 👨‍💼 Admin Interface

### Access the Admin Dashboard
Visit `/admin/dashboard` in your browser and enter your admin key.

### Admin API Endpoints

#### User Management
- `GET /admin/users` - List all users
- `POST /admin/users` - Create new user
- `GET /admin/users/{token}` - Get user details
- `PUT /admin/users/{token}` - Update user
- `POST /admin/users/{token}/disable` - Ban user
- `POST /admin/users/{token}/enable` - Unban user
- `POST /admin/users/{token}/rotate` - Rotate token
- `DELETE /admin/users/{token}` - Delete user

#### Statistics
- `GET /admin/stats` - System statistics
- `GET /admin/events` - Event logs

#### Bulk Operations
- `POST /admin/bulk/refresh-quotas` - Reset all quotas

### Create User via API
```bash
curl -X POST \
  -H "Authorization: Bearer your_admin_key" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "normal",
    "nickname": "John Doe",
    "token_limits": {
      "openai": 100000,
      "anthropic": 50000
    }
  }' \
  https://your-proxy.com/admin/users
```

## 📊 User Management Features

### User Types
- **Normal**: Standard users with default quotas
- **Special**: Premium users with higher limits and no rate limiting
- **Temporary**: Time-limited users (expire after 24 hours)

### User Properties
- **Token**: UUID-based authentication token
- **Nickname**: Human-readable identifier
- **Type**: User privilege level
- **IP Tracking**: Monitors IP addresses used
- **Usage Tracking**: Token consumption per model family
- **Quotas**: Configurable limits per model family
- **Ban System**: Disable with reasons

### Automatic Features
- **IP Limit Enforcement**: Auto-ban users with too many IPs
- **Quota Tracking**: Real-time token usage monitoring
- **Event Logging**: Comprehensive audit trail
- **Temporary User Cleanup**: Automatic expiration handling

## 🔄 Usage Tracking

### Token Counting
- Input tokens (prompt)
- Output tokens (completion)
- Total tokens
- Cost calculation in USD

### Per-Model Tracking
- OpenAI models (GPT-3.5, GPT-4, etc.)
- Anthropic models (Claude)
- Google AI models
- Mistral models

### Event Types
- `chat_completion`: API request completion
- `new_ip`: New IP address detected
- `user_action`: Administrative actions

## 🔥 Firebase Integration

### Setup Firebase
1. Create a Firebase project
2. Enable Realtime Database
3. Create service account with database access
4. Download service account key JSON

### Configuration
```bash
# Firebase URL
FIREBASE_URL=https://your-project-default-rtdb.firebaseio.com/

# Service account key (JSON string)
FIREBASE_SERVICE_ACCOUNT_KEY='{"type":"service_account",...}'
```

### Data Structure
```json
{
  "users": {
    "user_token_uuid": {
      "token": "uuid",
      "type": "normal",
      "created_at": "2024-01-01T00:00:00",
      "nickname": "John Doe",
      "ip": ["hashed_ip_1", "hashed_ip_2"],
      "token_counts": {
        "openai": {"input": 1000, "output": 500, "total": 1500},
        "anthropic": {"input": 800, "output": 400, "total": 1200}
      },
      "token_limits": {
        "openai": 100000,
        "anthropic": 50000
      },
      "disabled_at": null,
      "disabled_reason": null
    }
  }
}
```

## 🛡️ Security Features

### IP Privacy
- All IP addresses are SHA-256 hashed
- No raw IP storage
- Privacy-preserving tracking

### Token Security
- UUID-based tokens (unguessable)
- Token rotation capability
- Automatic cleanup

### Rate Limiting
- Per-user and per-IP limits
- Sliding window algorithm
- Configurable limits

### Admin Security
- Bearer token authentication
- Admin key rotation support
- Audit logging

## ⚙️ Configuration Reference

### Authentication
```bash
AUTH_MODE=user_token              # none, proxy_key, user_token
PROXY_PASSWORD=secure_password    # For proxy_key mode
ADMIN_KEY=admin_key              # Admin access key
```

### User Limits
```bash
MAX_IPS_PER_USER=3               # IP addresses per user
MAX_IPS_AUTO_BAN=true            # Auto-ban on IP limit
RATE_LIMIT_PER_MINUTE=60         # Requests per minute
```

### Firebase
```bash
FIREBASE_URL=https://...         # Firebase Realtime Database URL
FIREBASE_SERVICE_ACCOUNT_KEY={}  # Service account JSON
```

## 🔍 Monitoring & Analytics

### Dashboard Features
- Real-time user statistics
- Usage analytics
- Token consumption tracking
- Cost monitoring
- Event logging

### Admin Tools
- User creation and management
- Bulk operations
- Usage reports
- Event filtering

## 🚨 Troubleshooting

### Common Issues

1. **Firebase Connection Failed**
   - Check Firebase URL format
   - Verify service account permissions
   - Ensure database rules allow read/write

2. **Authentication Errors**
   - Verify AUTH_MODE setting
   - Check token format
   - Ensure admin key is correct

3. **Rate Limiting Issues**
   - Check RATE_LIMIT_PER_MINUTE setting
   - Verify user quotas
   - Monitor IP limits

### Debug Mode
Set `debug=True` in `app.run()` for detailed error messages.

## 📈 Performance Considerations

### Memory Usage
- In-memory user cache with Firebase sync
- Automatic cleanup of old data
- Efficient token counting

### Database Optimization
- Batch Firebase operations
- Background sync thread
- Minimal data structure

### Scalability
- Stateless authentication
- Horizontal scaling support
- Firebase handles concurrent access

## 🎯 Best Practices

1. **Regular Backups**: Export user data periodically
2. **Monitor Usage**: Track token consumption and costs
3. **Rotate Keys**: Regularly rotate admin and user tokens
4. **Audit Logs**: Review event logs for suspicious activity
5. **Quota Management**: Set appropriate limits for user types

## 📝 API Examples

### Create User
```python
import requests

response = requests.post('https://your-proxy.com/admin/users', 
    headers={'Authorization': 'Bearer admin_key'},
    json={
        'type': 'normal',
        'nickname': 'Test User',
        'token_limits': {'openai': 50000}
    })
```

### Check Usage
```python
response = requests.get('https://your-proxy.com/admin/users/user_token',
    headers={'Authorization': 'Bearer admin_key'})
```

### Disable User
```python
response = requests.post('https://your-proxy.com/admin/users/user_token/disable',
    headers={'Authorization': 'Bearer admin_key'},
    json={'reason': 'Abuse detected'})
```

This authentication system provides enterprise-grade user management with comprehensive tracking, Firebase integration, and powerful admin tools.