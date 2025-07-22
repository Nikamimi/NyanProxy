---
title: NyanProxy
emoji: 🐱
colorFrom: pink
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 🐱 NyanProxy - Meow-nificent AI Proxy!

A sophisticated AI proxy service that provides unified access to multiple AI providers with advanced features like dynamic model management, token tracking, rate limiting, and user management.

## 🔥 **IMPORTANT: Firebase Database Required!**

**NyanProxy requires Firebase Realtime Database for persistent logging and user management.** Without Firebase, the application will run with limited functionality (no logging, no user persistence, no analytics).

### **Why Firebase is Essential:**
- **Persistent User Data**: User tokens, quotas, and authentication
- **Event Logging**: All API requests, token usage, and analytics
- **Cross-Deployment Persistence**: Data survives app restarts and redeployments
- **Real-time Analytics**: Live usage monitoring and performance metrics
- **Admin Dashboard**: User management and system monitoring

## 🔐 **Privacy & Data Logging Policy**

**⚠️ IMPORTANT: Your prompts and responses are NOT logged!**

### **What IS logged:**
- ✅ **Usage metrics**: Request counts, response times, success/failure rates
- ✅ **Token counts**: Input/output token usage for billing and quotas
- ✅ **Model usage**: Which models are being used and how often
- ✅ **User activity**: Anonymous user tokens and authentication events
- ✅ **System health**: API key health, error rates, performance metrics
- ✅ **Timestamps**: When requests were made (not what was requested)

### **What is NOT logged:**
- ❌ **Prompt content**: Your actual questions, messages, or input text
- ❌ **Response content**: AI-generated responses or output text
- ❌ **Personal data**: No names, emails, or personally identifiable information
- ❌ **Conversation history**: No chat logs or message content
- ❌ **IP addresses**: Only hashed IP addresses for abuse prevention

**NyanProxy is designed to be a privacy-first proxy that tracks usage without compromising your data privacy.**

## ✨ Key Features

- **🧬 Dynamic Model Management**: Add/remove AI models via admin interface without code changes
- **🔥 Firebase-Powered**: Persistent logging and user management
- **🤖 Multi-Provider Support**: OpenAI, Anthropic, Google, Mistral, Groq, Cohere
- **🔄 Enhanced Retry Logic**: Intelligent key management with automatic failure recovery
- **🛡️ Smart Key Health Monitoring**: Real-time API key validation and removal
- **📊 Advanced Analytics**: Real-time usage tracking and cost analysis
- **👥 User Management**: Individual user tokens with quotas and permissions
- **🎯 Model Whitelisting**: Secure whitelist-only model access
- **📱 Beautiful Admin Dashboard**: Web interface for monitoring and management
- **🔐 Flexible Authentication**: Support for proxy keys or individual user tokens

## 📚 Documentation

### 📖 Detailed Guides
- **[🧬 Model Management](docs/model-management.md)** - Add/configure AI models dynamically
- **[🔐 Authentication Setup](docs/AUTHENTICATION.md)** - User management and authentication modes
- **[🔄 AI Key Management](docs/AI_KEY_MANAGEMENT_README.md)** - Enhanced retry logic and key health monitoring
- **[💬 Conversation Logging](docs/CONVERSATION_LOGGING.md)** - Privacy-focused usage tracking
- **[🔧 Integration Guide](docs/INTEGRATION_STEPS.md)** - Step-by-step implementation details

### 🎯 Quick Links
- **[💡 Usage Examples](#-usage-examples)** - API integration examples
- **[🔧 API Endpoints](#-api-endpoints)** - Complete endpoint reference
- **[⚙️ Environment Variables](#step-2-api-keys-setup)** - Configuration guide

*More documentation coming soon as features are added!*

## 🚀 Quick Setup Guide

### **Step 1: Firebase Database Setup (REQUIRED)**

1. **Create a Firebase Project**:
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project or use existing one
   - Enable **Realtime Database** (not Firestore)

2. **Get Firebase Configuration**:
   - Go to Project Settings → Service Accounts
   - Generate new private key (downloads JSON file)
   - Copy the JSON content

3. **Set Firebase Environment Variables**:
   ```bash
   FIREBASE_URL=https://your-project-default-rtdb.firebaseio.com
   FIREBASE_CREDENTIALS_JSON={"type":"service_account","project_id":"your-project",...}
   ```

### **Step 2: API Keys Setup**

**In Hugging Face Spaces Settings → Variables and Secrets:**

1. **AI Service API Keys**:
   ```bash
   OPENAI_API_KEYS=sk-key1,sk-key2,sk-key3
   ANTHROPIC_API_KEYS=sk-ant-key1,sk-ant-key2
   ```

2. **Firebase Configuration**:
   ```bash
   FIREBASE_URL=https://your-project-default-rtdb.firebaseio.com
   FIREBASE_CREDENTIALS_JSON={"type":"service_account",...}
   ```

3. **App Configuration**:
   ```bash
   AUTH_MODE=user_token
   FLASK_SECRET_KEY=your-secure-secret-key
   ```

### **Step 3: Local Development**

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your credentials**:
   ```bash
   # Required: Firebase Configuration
   FIREBASE_URL=https://your-project-default-rtdb.firebaseio.com
   FIREBASE_CREDENTIALS_JSON={"type":"service_account",...}
   
   # Required: AI Service API Keys
   OPENAI_API_KEYS=sk-key1,sk-key2,sk-key3
   ANTHROPIC_API_KEYS=sk-ant-key1,sk-ant-key2
   
   # Optional: Authentication
   AUTH_MODE=user_token
   FLASK_SECRET_KEY=your-secure-secret-key
   
   # Optional: Retry Configuration
   MAX_RETRIES=3
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally**:
   ```bash
   python app.py
   # or for Windows:
   start.bat
   ```

## 🔄 Enhanced Retry Logic & Key Management

NyanProxy features an intelligent retry system that automatically handles API key failures:

### **Automatic Key Health Monitoring**
- **Real-time validation**: Continuous monitoring of API key health
- **Provider-specific error handling**: Different rules for OpenAI, Anthropic, and Google
- **Intelligent key removal**: Invalid and quota-exceeded keys are permanently removed

### **Smart Retry Behavior**
- **Different keys per retry**: Each retry attempt uses a different API key
- **Permanent key removal**: Invalid keys are immediately removed from the pool
- **Provider-specific classification**:
  - **OpenAI**: 401=Invalid, 429=Quota Exceeded (both removed)
  - **Anthropic**: 401=Invalid, 429=Rate Limited (kept), 403=Quota Exceeded (removed)
  - **Google**: 403=Invalid (removed), 429=Rate Limited (kept)

### **Dashboard Integration**
- **Live key status**: See healthy, rate limited, invalid, and quota exceeded counts
- **Key identification**: Shows first 8 and last 4 characters for easy identification
- **Real-time updates**: Dashboard reflects current key pool status

### **Configuration**
```bash
# Set maximum retry attempts (default: 3)
MAX_RETRIES=3

# Multiple keys for automatic failover
ANTHROPIC_API_KEYS=sk-ant-key1,sk-ant-key2,sk-ant-key3
```

## 🚀 API Endpoints

### **Core Endpoints**
- `GET /` - 📊 **Dashboard** - Beautiful web interface with real-time metrics
- `GET /health` - ❤️ **Health Check** - System status and API key health
- `POST /v1/chat/completions` - 🤖 **OpenAI Chat** - Compatible with OpenAI API
- `POST /v1/messages` - 🤖 **Anthropic Messages** - Compatible with Anthropic API
- `GET /v1/models` - 📋 **Model List** - Available whitelisted models

### **Management Endpoints**
- `GET /admin` - 👑 **Admin Dashboard** - User and system management
- `GET /api/metrics` - 📊 **Real-time Metrics** - Usage statistics and performance
- `GET /api/keys/status` - 🔑 **API Key Status** - Health of configured keys
- `GET /api/keys/health` - 💚 **Detailed Health** - Comprehensive key diagnostics

## 💡 Usage Examples

### **OpenAI Compatible**
```python
import openai

client = openai.OpenAI(
    base_url="https://your-nyanproxy.hf.space/v1",
    api_key="your-user-token"  # Individual user token
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### **Anthropic Compatible**
```python
import anthropic

client = anthropic.Anthropic(
    base_url="https://your-nyanproxy.hf.space/v1",
    api_key="your-user-token"
)

response = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## ⚠️ **Without Firebase: Limited Mode**

If Firebase is not configured, NyanProxy will run in **limited mode**:
- ❌ No user persistence (users lost on restart)
- ❌ No event logging (no usage tracking)
- ❌ No analytics (no dashboard metrics)
- ❌ No admin features (no user management)
- ✅ Basic proxying still works (but without persistence)

**For production use, Firebase is absolutely required!**
