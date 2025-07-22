# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Primary method - Production with graceful shutdown
python run.py

# Hugging Face compatible entry point
python app.py

# Windows development (uses hardcoded Python path)
start.bat
```

### Key Environment Variables
- `PORT=7860` - Default port for the Flask application
- `DEBUG=true/false` - Enables/disables debug mode
- `FIREBASE_URL` - Required Firebase Realtime Database URL
- `FIREBASE_CREDENTIALS_JSON` - Required Firebase service account credentials
- `AUTH_MODE` - Authentication mode (user_token, etc.)
- AI API keys: `OPENAI_API_KEYS`, `ANTHROPIC_API_KEYS`, etc.

### Installation
```bash
pip install -r requirements.txt
```

## Architecture Overview

### Core Application Structure
- **Entry Points**: 
  - `app.py` - Hugging Face compatible entry point
  - `run.py` - Primary startup script with graceful shutdown handling
  - `core/app.py` - Main Flask application with all route definitions

### Key Components

#### Authentication & User Management
- **`src/middleware/auth.py`** - Authentication middleware with quota tracking
- **`src/config/auth.py`** - Authentication configuration
- **`src/services/user_store.py`** - User management with Firebase integration

#### AI Provider Integration
- **`src/services/model_families.py`** - Dynamic model management system with whitelisting
- **`ai/tokenizers/`** - Unified tokenization for different AI providers
  - Supports OpenAI, Anthropic, Mistral with fallback counting

#### Logging & Analytics
- **`src/services/firebase_logger.py`** - Firebase-backed event logging (privacy-focused)
- **`src/services/conversation_logger.py`** - Request/response tracking (no content logging)

#### Admin Interface
- **`src/routes/admin.py`** - API endpoints for admin functionality
- **`src/routes/admin_web.py`** - Web-based admin dashboard
- **`src/routes/model_families_admin.py`** - Model management admin interface

### Key Features

#### Privacy-First Logging
The system logs usage metrics but never logs prompt/response content. Only metadata like token counts, model usage, and response times are recorded.

#### Dynamic Model Management
Models can be added/removed via admin interface without code changes. The `model_families.py` system handles whitelisting and provider routing.

#### Multi-Provider Support
Unified interface for OpenAI, Anthropic, Google, Mistral, Groq, and Cohere APIs with automatic key rotation and health monitoring.

#### Firebase Integration
- **Required for production** - User persistence, logging, analytics
- Graceful degradation without Firebase (limited functionality)
- Real-time database for user management and event tracking

### Important Implementation Notes

#### Shutdown Handling
`run.py` implements comprehensive graceful shutdown with:
- Thread manager cleanup
- Connection pool closure
- Firebase logging of shutdown events
- Garbage collection and resource cleanup

#### Health Monitoring
`core/health_checker.py` provides API key health monitoring with automatic rotation and failure detection.

#### Token Counting
Unified tokenizer system handles different provider tokenization methods with intelligent fallbacks when specific tokenizers aren't available.

## Firebase Dependency

This application **requires Firebase Realtime Database** for:
- User authentication and quota management
- Usage analytics and logging
- Admin dashboard functionality
- Cross-deployment data persistence

Without Firebase, the app runs in limited mode with no persistence or logging capabilities.