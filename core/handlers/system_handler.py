"""
System Route Handler for NyanProxy

Handles system metrics, monitoring, and diagnostic endpoints
Extracted from core/app.py for better modularity
"""
import time
from datetime import datetime
from typing import Dict, Any, Tuple
from flask import jsonify

from .base_handler import BaseHandler


class SystemHandler(BaseHandler):
    """Handler for system monitoring and metrics routes"""
    
    def get_metrics(self) -> Tuple[Any, int]:
        """Get comprehensive proxy metrics"""
        start_time = self.track_request_start()
        
        try:
            metrics_data = self.metrics.get_metrics()
            self.track_request_end('metrics', start_time)
            return self.create_success_response(metrics_data), 200
            
        except Exception as e:
            self.track_request_end('metrics', start_time, error=True)
            return self.create_error_response(str(e), 500)
    
    def system_health(self) -> Tuple[Any, int]:
        """Comprehensive system health endpoint with warnings and recommendations"""
        start_time = self.track_request_start()
        
        try:
            health_data = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': self.metrics.get_uptime(),
                'metrics': self.metrics.get_metrics(),
                'memory_warning': False,
                'thread_warning': False,
                'recommendations': []
            }
            
            system_stats = health_data['metrics']['system_stats']
            thread_stats = health_data['metrics']['thread_stats']
            
            # Check for warnings
            if system_stats['memory_mb'] > 400:
                health_data['memory_warning'] = True
                health_data['recommendations'].append('Consider reducing memory usage or restarting')
                
            if system_stats['thread_count'] > 50:
                health_data['thread_warning'] = True
                health_data['recommendations'].append('High thread count detected - monitor for thread leaks')
            
            # Check for restarts
            if health_data['metrics']['restart_count'] > 5:
                health_data['recommendations'].append('Frequent restarts detected - investigate stability issues')
            
            # Force garbage collection if needed
            memory_check, collected = self.metrics.check_memory_threshold(400)
            if memory_check:
                health_data['recommendations'].append(f'Forced garbage collection - {collected} objects collected')
            
            self.track_request_end('system_health', start_time)
            return self.create_success_response(health_data), 200
            
        except Exception as e:
            self.track_request_end('system_health', start_time, error=True)
            health_data = {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'recommendations': ['System health check failed - investigate immediately']
            }
            return self.create_success_response(health_data), 500