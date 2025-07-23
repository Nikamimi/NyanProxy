"""
Refactored NyanProxy Flask Application

This is the cleaned up version of core/app.py with extracted services
- ThreadManager extracted to core/services/thread_manager.py
- ConnectionPoolManager extracted to core/services/connection_pool.py  
- MetricsTracker extracted to core/services/metrics_tracker.py
- APIKeyManager extracted to core/services/legacy_api_key_manager.py
"""
from flask import Flask, request, jsonify, Response, render_template, g, session
import requests
import os
import json
import random
import time
import threading
import hashlib
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .health_checker import health_manager, HealthResult

# Import our extracted services
from .services import ThreadManager, ConnectionPoolManager, MetricsTracker
from .services.legacy_api_key_manager import LegacyAPIKeyManager

# Import tokenizer libraries
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not available, token counting will be approximate")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    import os
    # Load .env from the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, '.env')
    load_dotenv(env_path)
    # Loading .env file
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

from ai.tokenizers.unified_tokenizer import unified_tokenizer

# Import authentication system
from src.config.auth import auth_config
from src.middleware.auth import require_auth, check_quota, track_token_usage, get_client_ip, authenticate_request
from src.services.firebase_logger import event_logger, structured_logger
from src.services.user_store import user_store, AuthResult
from src.routes.admin import admin_bp
from src.routes.admin_web import admin_web_bp
from src.routes.model_families_admin import model_families_bp
from src.middleware.auth import generate_csrf_token
from src.services.model_families import model_manager, AIProvider
from src.services.conversation_logger import conversation_logger

# Initialize our extracted services
thread_manager = ThreadManager()
connection_pool = ConnectionPoolManager()
key_manager = LegacyAPIKeyManager()
metrics = MetricsTracker(thread_manager)  # Inject thread_manager dependency

# Flask app initialization
app = Flask(__name__, template_folder='../pages', static_folder='../static')

# Startup logging and restart detection
startup_time = datetime.now()
print(f"[CAT] NyanProxy starting up at {startup_time.isoformat()}")
print(f"[TOOL] Process ID: {os.getpid()}")
print(f"[THREAD] Main thread ID: {threading.current_thread().ident}")

# Initialize conversation logger
try:
    import sys
    print(f"[PYTHON] Python executable: {sys.executable}")
    print(f"[PYTHON] Python version: {sys.version}")
    print(f"[PYTHON] Python path: {sys.path[:3]}...")  # Show first 3 paths
    
    # Test imports
    try:
        import gspread
        print(f"[OK] gspread imported successfully - version: {gspread.__version__}")
    except ImportError as e:
        print(f"[FAIL] Failed to import gspread: {e}")
    
    try:
        from google.oauth2.service_account import Credentials
        print(f"[OK] google-auth imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import google-auth: {e}")
    
    conversation_logger.reconnect_firebase()
    print(f" Conversation logger initialized - Enabled: {conversation_logger.enabled}")
except Exception as e:
    print(f"[WARN] Warning: Failed to initialize conversation logger: {e}")
    import traceback
    traceback.print_exc()

# Check for previous crash/restart indicators
restart_file = os.path.join(os.path.dirname(__file__), '..', '.restart_count')
try:
    if os.path.exists(restart_file):
        with open(restart_file, 'r') as f:
            restart_count = int(f.read().strip()) + 1
            metrics.restart_count = restart_count
        print(f"[RETRY] Detected restart #{restart_count}")
    else:
        restart_count = 1
        metrics.restart_count = restart_count
        print(" First startup detected")
    
    # Update restart count file
    with open(restart_file, 'w') as f:
        f.write(str(restart_count))
    
    # Log startup event to structured logger if available
    try:
        structured_logger.log_system_event(
            event_type='startup',
            details={
                'restart_count': restart_count,
                'startup_time': startup_time.isoformat(),
                'process_id': os.getpid(),
                'thread_id': threading.current_thread().ident
            }
        )
    except Exception as e:
        print(f"[WARN] Could not log startup event: {e}")
        
except Exception as e:
    print(f"[WARN] Error handling restart detection: {e}")
    metrics.restart_count = 1

# Configure Flask
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key-change-in-production')
app.permanent_session_lifetime = timedelta(hours=24)  # Sessions last 24 hours

# Register admin blueprints
app.register_blueprint(admin_bp)
app.register_blueprint(admin_web_bp)
app.register_blueprint(model_families_bp)

# Add CSRF token to template context
@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf_token)

# Load anti-abuse configuration from Firebase at startup
try:
    from src.services.config_manager import config_manager
    anti_abuse_config = config_manager.load_anti_abuse_config()
    
    # Update auth_config with persisted Firebase values
    auth_config.rate_limit_enabled = anti_abuse_config.rate_limit_enabled
    auth_config.rate_limit_per_minute = anti_abuse_config.rate_limit_per_minute
    auth_config.max_ips_per_user = anti_abuse_config.max_ips_per_user
    auth_config.max_ips_auto_ban = anti_abuse_config.max_ips_auto_ban
    
    print(f"[CAT] STARTUP: Loaded anti-abuse config - max_ips_per_user: {auth_config.max_ips_per_user}")
    print(f"[CAT] STARTUP: Loaded anti-abuse config - max_ips_auto_ban: {auth_config.max_ips_auto_ban}")
except Exception as e:
    print(f"[ERROR] STARTUP: Failed to load anti-abuse config: {e}")
    print(f"[CAT] STARTUP: Using defaults - max_ips_per_user: {auth_config.max_ips_per_user}")

# TODO: Replace with sophisticated health check system
# Run initial health checks in background using our extracted services
def run_initial_health_checks():
    """Run initial health checks for all keys concurrently in background"""
    def check_all_keys():
        # Collect all key-service pairs for concurrent execution
        key_service_pairs = []
        for service, keys in key_manager.api_keys.items():
            for key in keys:
                if key:
                    key_service_pairs.append((service, key))
        
        if not key_service_pairs:
            return
        
        print(f"[ROCKET] Starting concurrent health checks for {len(key_service_pairs)} API keys...")
        start_time = time.time()
        
        # Use managed thread pool for health checks
        health_pool = thread_manager.create_thread_pool("health_checks", max_workers=min(30, len(key_service_pairs)))
        
        # Submit all health check tasks
        futures = []
        for service, key in key_service_pairs:
            future = health_pool.submit(key_manager.perform_proactive_health_check, service, key)
            futures.append((future, service, key))
        
        # Process completed tasks
        completed = 0
        for future, service, key in futures:
            try:
                future.result(timeout=30)  # Add timeout
                completed += 1
                
                # Log progress every 20 completions or when significant milestones reached
                if completed % 20 == 0 or completed in [10, 25, 50, 100]:
                    elapsed_partial = time.time() - start_time
                    rate = completed / elapsed_partial if elapsed_partial > 0 else 0
                    print(f"[OK] Health checks: {completed}/{len(key_service_pairs)} ({rate:.1f}/sec)")
                    
            except Exception as e:
                print(f"[WARN] Health check failed for {service} key {key[:8]}...: {e}")
                completed += 1
        
        elapsed = time.time() - start_time
        print(f"[SUCCESS] All health checks completed in {elapsed:.2f}s ({len(key_service_pairs)/elapsed:.1f} keys/sec)")
    
    # Use managed daemon thread
    thread_manager.create_daemon_thread(check_all_keys, "health_check_runner")

# Run the health checks
run_initial_health_checks()

# Define the base URL helper (needed for dashboard)
def get_base_url():
    """Get the base URL for the current request"""
    if request:
        return request.host_url.rstrip('/')
    else:
        # Fallback for non-request contexts
        return os.getenv('PUBLIC_URL', 'http://localhost:7860')

# Request logging middleware removed for cleaner output

@app.route('/health', methods=['GET'])
def health_check():
    start_time = time.time()
    result = jsonify({"status": "healthy", "service": "AI Proxy"})
    metrics.track_request('health', time.time() - start_time)
    return result

# NOTE: The rest of the routes would continue here...
# For now I'm showing the refactored structure with extracted services
# The actual route implementations would remain the same but use the extracted services

if __name__ == '__main__':
    print("[TARGET] Refactored NyanProxy loaded with extracted services!")
    print("[CHART] Services initialized:")
    print(f"   - ThreadManager: {thread_manager}")
    print(f"   - ConnectionPoolManager: {connection_pool}")
    print(f"   - MetricsTracker: {metrics}")
    print(f"   - LegacyAPIKeyManager: {key_manager}")