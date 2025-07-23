"""
 Firebase-based Event Logging System for NyanProxy
===================================================

This module provides a comprehensive event logging system using Firebase Realtime Database
for persistent, scalable logging across deployments.

 PRIVACY NOTICE:
- This logger only tracks USAGE METRICS and ANALYTICS
- NO PROMPT CONTENT or RESPONSE CONTENT is ever logged
- Only metadata like token counts, timestamps, and model usage is stored
- User privacy is protected while maintaining usage tracking for quotas and billing
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib

try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: Firebase not available. Logging will be disabled.")

@dataclass
class Event:
    """Simple event structure for basic logging"""
    token: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EventType(Enum):
    """Event types for structured logging"""
    CHAT_COMPLETION = "chat_completion"
    STREAMING_REQUEST = "streaming_request"
    MODEL_LIST_REQUEST = "model_list_request"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    RATE_LIMIT_HIT = "rate_limit_hit"
    API_ERROR = "api_error"
    HEALTH_CHECK = "health_check"

class FirebaseEventLogger:
    """
    Firebase-based event logger for persistent logging
    """
    
    def __init__(self, firebase_path: str = "events"):
        self.firebase_path = firebase_path
        self.lock = threading.Lock()
        self.firebase_available = FIREBASE_AVAILABLE
        
        if not self.firebase_available:
            print("Warning: Firebase not available. Events will not be logged.")
            return
            
        try:
            # Test Firebase connection
            self.db_ref = db.reference(self.firebase_path)
            self.db_ref.child("test").set({"initialized": True, "timestamp": datetime.now().isoformat()})
            # Firebase logging initialized
        except Exception as e:
            print(f"Warning: Firebase logging failed to initialize: {e}")
            self.firebase_available = False
    
    def log_event(self, event: Event) -> bool:
        """Log an event to Firebase"""
        if not self.firebase_available:
            return False
            
        try:
            with self.lock:
                event_data = {
                    "token": event.token,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "timestamp": event.timestamp.isoformat() if event.timestamp else datetime.now().isoformat()
                }
                
                # Use timestamp-based key for ordering
                timestamp_key = int(time.time() * 1000)  # milliseconds
                event_key = f"{timestamp_key}_{uuid.uuid4().hex[:8]}"
                
                self.db_ref.child(event_key).set(event_data)
                return True
                
        except Exception as e:
            print(f"Failed to log event to Firebase: {e}")
            return False
    
    def log_chat_completion(self, token: str, model_family: str, input_tokens: int, 
                           output_tokens: int, ip_hash: str = "", success: bool = True) -> bool:
        """Log a chat completion event"""
        event = Event(
            token=token,
            event_type="chat_completion",
            payload={
                "model_family": model_family,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "ip_hash": ip_hash,
                "success": success
            }
        )
        return self.log_event(event)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, 
                       response_time: float, token: str = None) -> bool:
        """Log an API request event"""
        event = Event(
            token=token or "anonymous",
            event_type="api_request",
            payload={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time": response_time,
                "success": status_code < 400
            }
        )
        return self.log_event(event)
    
    def get_user_stats(self, token: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        if not self.firebase_available:
            return {}
            
        try:
            # Get events from last N days
            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            events = self.db_ref.order_by_key().start_at(str(cutoff_timestamp)).get()
            
            if not events:
                return {}
            
            stats = {
                "total_requests": 0,
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "chat_completions": 0,
                "api_errors": 0,
                "model_families": {}
            }
            
            for event_key, event_data in events.items():
                if event_data.get("token") == token:
                    payload = event_data.get("payload", {})
                    event_type = event_data.get("event_type")
                    
                    stats["total_requests"] += 1
                    
                    if event_type == "chat_completion":
                        stats["chat_completions"] += 1
                        stats["input_tokens"] += payload.get("input_tokens", 0)
                        stats["output_tokens"] += payload.get("output_tokens", 0)
                        stats["total_tokens"] += payload.get("total_tokens", 0)
                        
                        model_family = payload.get("model_family", "unknown")
                        if model_family not in stats["model_families"]:
                            stats["model_families"][model_family] = 0
                        stats["model_families"][model_family] += 1
                    
                    elif event_type == "api_request" and not payload.get("success", True):
                        stats["api_errors"] += 1
            
            return stats
            
        except Exception as e:
            print(f"Failed to get user stats: {e}")
            return {}
    
    def log_user_action(self, token: str, action: str, details: Dict[str, Any] = None, 
                       admin_user: str = None) -> bool:
        """Log a user action event (admin actions, etc.)"""
        event = Event(
            token=token,
            event_type="user_action",
            payload={
                "action": action,
                "details": details or {},
                "admin_user": admin_user
            }
        )
        return self.log_event(event)

    def cleanup_old_events(self, days: int = 90) -> bool:
        """Clean up events older than specified days"""
        if not self.firebase_available:
            return False
            
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            # Get old events
            old_events = self.db_ref.order_by_key().end_at(str(cutoff_timestamp)).get()
            
            if old_events:
                # Delete old events in batches
                batch_size = 100
                deleted_count = 0
                
                for event_key in list(old_events.keys()):
                    self.db_ref.child(event_key).delete()
                    deleted_count += 1
                    
                    if deleted_count % batch_size == 0:
                        time.sleep(0.1)  # Small delay to avoid rate limits
                
                # Cleaned up old events from Firebase
                return True
            
            return True
            
        except Exception as e:
            print(f"Failed to cleanup old events: {e}")
            return False

class FirebaseStructuredLogger:
    """
    Advanced Firebase-based structured logger with detailed analytics
    """
    
    def __init__(self, firebase_path: str = "structured_events"):
        self.firebase_path = firebase_path
        self.lock = threading.Lock()
        self.firebase_available = FIREBASE_AVAILABLE
        
        if not self.firebase_available:
            print("Warning: Firebase not available. Structured logging will be disabled.")
            return
            
        try:
            # Test Firebase connection
            self.db_ref = db.reference(self.firebase_path)
            self.db_ref.child("test").set({"initialized": True, "timestamp": datetime.now().isoformat()})
            # Firebase structured logging initialized
        except Exception as e:
            print(f"Warning: Firebase structured logging failed to initialize: {e}")
            self.firebase_available = False
    
    def log_chat_completion(self, user_token: str, model_family: str, model_name: str,
                           input_tokens: int, output_tokens: int, cost_usd: float,
                           response_time_ms: float, success: bool, ip_hash: str = "",
                           user_agent: str = "") -> bool:
        """Log a detailed chat completion event"""
        if not self.firebase_available:
            return False
            
        try:
            with self.lock:
                event_data = {
                    "event_type": "chat_completion",
                    "timestamp": datetime.now().isoformat(),
                    "user_token": user_token,
                    "model_family": model_family,
                    "model_name": model_name,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": cost_usd,
                    "response_time_ms": response_time_ms,
                    "success": success,
                    "ip_hash": ip_hash,
                    "user_agent": user_agent
                }
                
                # Use timestamp-based key for ordering
                timestamp_key = int(time.time() * 1000)  # milliseconds
                event_key = f"{timestamp_key}_{uuid.uuid4().hex[:8]}"
                
                self.db_ref.child(event_key).set(event_data)
                return True
                
        except Exception as e:
            print(f"Failed to log structured event to Firebase: {e}")
            return False
    
    def log_user_action(self, user_token: str, action: str, details: Dict[str, Any] = None, 
                       admin_user: str = None) -> bool:
        """Log a user action event (admin actions, etc.)"""
        if not self.firebase_available:
            return False
            
        try:
            with self.lock:
                event_data = {
                    "event_type": "user_action",
                    "timestamp": datetime.now().isoformat(),
                    "user_token": user_token,
                    "action": action,
                    "details": details or {},
                    "admin_user": admin_user
                }
                
                # Use timestamp-based key for ordering
                timestamp_key = int(time.time() * 1000)  # milliseconds
                event_key = f"{timestamp_key}_{uuid.uuid4().hex[:8]}"
                
                self.db_ref.child(event_key).set(event_data)
                return True
                
        except Exception as e:
            print(f"Failed to log user action to Firebase: {e}")
            return False
    
    def get_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics from Firebase"""
        if not self.firebase_available:
            return {}
            
        try:
            # Get events from last N days
            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            events = self.db_ref.order_by_key().start_at(str(cutoff_timestamp)).get()
            
            if not events:
                return {}
            
            analytics = {
                "total_requests": len(events),
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0,
                "success_rate": 0.0,
                "model_usage": {},
                "user_activity": {},
                "hourly_distribution": {}
            }
            
            successful_requests = 0
            total_response_time = 0
            
            for event_key, event_data in events.items():
                if event_data.get("event_type") == "chat_completion":
                    analytics["total_tokens"] += event_data.get("total_tokens", 0)
                    analytics["total_cost"] += event_data.get("cost_usd", 0)
                    
                    if event_data.get("success", False):
                        successful_requests += 1
                        total_response_time += event_data.get("response_time_ms", 0)
                    
                    # Model usage
                    model = event_data.get("model_name", "unknown")
                    if model not in analytics["model_usage"]:
                        analytics["model_usage"][model] = 0
                    analytics["model_usage"][model] += 1
                    
                    # User activity
                    user = event_data.get("user_token", "anonymous")
                    if user not in analytics["user_activity"]:
                        analytics["user_activity"][user] = 0
                    analytics["user_activity"][user] += 1
                    
                    # Hourly distribution
                    timestamp = event_data.get("timestamp", "")
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            hour = dt.hour
                            if hour not in analytics["hourly_distribution"]:
                                analytics["hourly_distribution"][hour] = 0
                            analytics["hourly_distribution"][hour] += 1
                        except:
                            pass
            
            # Calculate averages
            if successful_requests > 0:
                analytics["avg_response_time"] = total_response_time / successful_requests
                analytics["success_rate"] = successful_requests / len(events)
            
            return analytics
            
        except Exception as e:
            print(f"Failed to get analytics: {e}")
            return {}
    
    def get_user_analytics(self, user_token: str, days: int = 30) -> Dict[str, Any]:
        """Get detailed analytics for a specific user"""
        if not self.firebase_available:
            return {}
            
        try:
            # Get events from last N days
            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            events = self.db_ref.order_by_key().start_at(str(cutoff_timestamp)).get()
            
            if not events:
                return {}
            
            # Filter events for this user
            user_events = {}
            for event_key, event_data in events.items():
                if event_data.get("user_token") == user_token:
                    user_events[event_key] = event_data
            
            if not user_events:
                return {}
            
            analytics = {
                "total_requests": len(user_events),
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0,
                "success_rate": 0.0,
                "models_used": [],
                "model_usage": {},
                "hourly_distribution": {},
                "daily_usage": {},
                "input_tokens": 0,
                "output_tokens": 0,
                "chat_completions": 0,
                "api_errors": 0
            }
            
            successful_requests = 0
            total_response_time = 0
            models_set = set()
            
            for event_key, event_data in user_events.items():
                if event_data.get("event_type") == "chat_completion":
                    analytics["chat_completions"] += 1
                    analytics["total_tokens"] += event_data.get("total_tokens", 0)
                    analytics["input_tokens"] += event_data.get("input_tokens", 0)
                    analytics["output_tokens"] += event_data.get("output_tokens", 0)
                    analytics["total_cost"] += event_data.get("cost_usd", 0)
                    
                    if event_data.get("success", False):
                        successful_requests += 1
                        total_response_time += event_data.get("response_time_ms", 0)
                    else:
                        analytics["api_errors"] += 1
                    
                    # Model usage
                    model = event_data.get("model_name", "unknown")
                    models_set.add(model)
                    if model not in analytics["model_usage"]:
                        analytics["model_usage"][model] = {
                            "requests": 0,
                            "tokens": 0,
                            "cost": 0.0
                        }
                    analytics["model_usage"][model]["requests"] += 1
                    analytics["model_usage"][model]["tokens"] += event_data.get("total_tokens", 0)
                    analytics["model_usage"][model]["cost"] += event_data.get("cost_usd", 0)
                    
                    # Hourly distribution
                    timestamp = event_data.get("timestamp", "")
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                            hour = dt.hour
                            date_str = dt.strftime("%Y-%m-%d")
                            
                            if hour not in analytics["hourly_distribution"]:
                                analytics["hourly_distribution"][hour] = 0
                            analytics["hourly_distribution"][hour] += 1
                            
                            if date_str not in analytics["daily_usage"]:
                                analytics["daily_usage"][date_str] = {
                                    "requests": 0,
                                    "tokens": 0,
                                    "cost": 0.0
                                }
                            analytics["daily_usage"][date_str]["requests"] += 1
                            analytics["daily_usage"][date_str]["tokens"] += event_data.get("total_tokens", 0)
                            analytics["daily_usage"][date_str]["cost"] += event_data.get("cost_usd", 0)
                        except:
                            pass
            
            # Calculate averages and final stats
            analytics["models_used"] = list(models_set)
            if successful_requests > 0:
                analytics["avg_response_time"] = total_response_time / successful_requests
                analytics["success_rate"] = (successful_requests / len(user_events)) * 100
            
            return analytics
            
        except Exception as e:
            print(f"Failed to get user analytics: {e}")
            return {}

# Global logger instances
firebase_event_logger = FirebaseEventLogger()
firebase_structured_logger = FirebaseStructuredLogger()

# Backward compatibility - use Firebase loggers if available, otherwise create dummy loggers
if FIREBASE_AVAILABLE:
    # Replace the old loggers with Firebase ones
    event_logger = firebase_event_logger
    structured_logger = firebase_structured_logger
else:
    # Create dummy loggers that don't crash the app
    class DummyLogger:
        def log_event(self, *args, **kwargs): return False
        def log_chat_completion(self, *args, **kwargs): return False
        def log_api_request(self, *args, **kwargs): return False
        def log_user_action(self, *args, **kwargs): return False
        def get_user_stats(self, *args, **kwargs): return {}
        def get_user_analytics(self, *args, **kwargs): return {}
        def cleanup_old_events(self, *args, **kwargs): return False
        def get_analytics(self, *args, **kwargs): return {}
    
    event_logger = DummyLogger()
    structured_logger = DummyLogger()