from flask import Blueprint, request, jsonify, g
from typing import Dict, Any, List
from datetime import datetime

from ..middleware.auth import require_admin_auth, require_auth
from ..services.user_store import user_store, UserType
from ..services.firebase_logger import event_logger

admin_bp = Blueprint('admin_api', __name__, url_prefix='/admin/api')

@admin_bp.route('/users', methods=['GET'])
@require_admin_auth
def list_users():
    """List all users with pagination and filtering"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 50))
    user_type = request.args.get('type')
    search = request.args.get('search', '').lower()
    
    all_users = user_store.get_all_users()
    
    # Apply filters
    filtered_users = []
    for user in all_users:
        if user_type and user.type.value != user_type:
            continue
        
        if search:
            if (search in user.token.lower() or 
                (user.nickname and search in user.nickname.lower()) or
                (user.disabled_reason and search in user.disabled_reason.lower())):
                filtered_users.append(user)
        else:
            filtered_users.append(user)
    
    # Pagination
    start = (page - 1) * limit
    end = start + limit
    paginated_users = filtered_users[start:end]
    
    # Convert to dict with additional stats
    users_data = []
    for user in paginated_users:
        user_dict = user.to_dict()
        
        # Add usage statistics
        total_tokens = sum(count.total for count in user.token_counts.values())
        total_requests = sum(usage.prompt_count for usage in user.ip_usage)
        
        user_dict['stats'] = {
            'total_tokens': total_tokens,
            'total_requests': total_requests,
            'ip_count': len(user.ip),
            'days_since_created': (datetime.now() - user.created_at).days
        }
        
        users_data.append(user_dict)
    
    return jsonify({
        'users': users_data,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': len(filtered_users),
            'has_next': end < len(filtered_users)
        }
    })

@admin_bp.route('/users', methods=['POST'])
@require_admin_auth
def create_user():
    """Create a new user"""
    data = request.get_json()
    
    user_type = UserType(data.get('type', 'normal'))
    nickname = data.get('nickname')
    token_limits = data.get('token_limits', {})
    
    try:
        token = user_store.create_user(
            user_type=user_type,
            token_limits=token_limits,
            nickname=nickname
        )
        
        # Log admin action
        event_logger.log_user_action(
            token=token,
            action='user_created',
            details={
                'created_by': 'admin',
                'type': user_type.value,
                'nickname': nickname
            }
        )
        
        return jsonify({
            'success': True,
            'token': token,
            'message': 'User created successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@admin_bp.route('/users/<token>', methods=['GET'])
@require_admin_auth
def get_user(token: str):
    """Get detailed user information"""
    user = user_store.get_user(token)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    user_dict = user.to_dict()
    
    # Add detailed statistics
    usage_stats = event_logger.get_usage_stats(token)
    user_dict['usage_stats'] = usage_stats
    
    # Add recent events
    recent_events = event_logger.get_events(token=token, limit=10)
    user_dict['recent_events'] = recent_events
    
    return jsonify(user_dict)

@admin_bp.route('/users/<token>', methods=['PUT'])
@require_admin_auth
def update_user(token: str):
    """Update user properties"""
    user = user_store.get_user(token)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    
    # Update nickname
    if 'nickname' in data:
        user.nickname = data['nickname']
    
    # Update token limits
    if 'token_limits' in data:
        user.token_limits.update(data['token_limits'])
    
    # Update user type
    if 'type' in data:
        user.type = UserType(data['type'])
    
    # Add to flush queue
    user_store.flush_queue.add(token)
    
    # Log admin action
    event_logger.log_user_action(
        token=token,
        action='user_updated',
        details={
            'updated_by': 'admin',
            'changes': data
        }
    )
    
    return jsonify({
        'success': True,
        'message': 'User updated successfully'
    })

@admin_bp.route('/users/<token>/disable', methods=['POST'])
@require_admin_auth
def disable_user(token: str):
    """Disable/ban a user"""
    data = request.get_json() or {}
    reason = data.get('reason', 'Disabled by admin')
    
    success = user_store.disable_user(token, reason)
    
    if not success:
        return jsonify({'error': 'User not found'}), 404
    
    # Log admin action
    event_logger.log_user_action(
        token=token,
        action='user_disabled',
        details={
            'disabled_by': 'admin',
            'reason': reason
        }
    )
    
    return jsonify({
        'success': True,
        'message': 'User disabled successfully'
    })

@admin_bp.route('/users/<token>/enable', methods=['POST'])
@require_admin_auth
def enable_user(token: str):
    """Enable/unban a user"""
    success = user_store.reactivate_user(token)
    
    if not success:
        return jsonify({'error': 'User not found'}), 404
    
    # Log admin action
    event_logger.log_user_action(
        token=token,
        action='user_enabled',
        details={
            'enabled_by': 'admin'
        }
    )
    
    return jsonify({
        'success': True,
        'message': 'User enabled successfully'
    })

@admin_bp.route('/users/<token>/rotate', methods=['POST'])
@require_admin_auth
def rotate_user_token(token: str):
    """Rotate user token"""
    new_token = user_store.rotate_user_token(token)
    
    if not new_token:
        return jsonify({'error': 'User not found'}), 404
    
    # Log admin action
    event_logger.log_user_action(
        token=new_token,
        action='token_rotated',
        details={
            'rotated_by': 'admin',
            'old_token': token[:8] + '...'
        }
    )
    
    return jsonify({
        'success': True,
        'new_token': new_token,
        'message': 'Token rotated successfully'
    })

@admin_bp.route('/users/<token>', methods=['DELETE'])
@require_admin_auth
def delete_user(token: str):
    """Delete a user (automatically disables first if needed)"""
    try:
        user = user_store.get_user(token)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # If user is not disabled, disable them first
        if not user.is_disabled():
            user_store.disable_user(token, "Disabled before deletion by admin")
            
            # Log disable action
            event_logger.log_user_action(
                token=token,
                action='user_disabled',
                details={
                    'disabled_by': 'admin',
                    'reason': 'Disabled before deletion by admin'
                }
            )
        
        # Now delete the user
        success = user_store.delete_user(token)
        
        if not success:
            return jsonify({'error': 'Failed to delete user'}), 500
        
        # Log admin action
        event_logger.log_user_action(
            token=token,
            action='user_deleted',
            details={
                'deleted_by': 'admin'
            }
        )
        
        return jsonify({
            'success': True,
            'message': 'User deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to delete user: {str(e)}'
        }), 500

@admin_bp.route('/stats', methods=['GET'])
@require_admin_auth
def get_admin_stats():
    """Get overall system statistics"""
    all_users = user_store.get_all_users()
    
    stats = {
        'total_users': len(all_users),
        'active_users': len([u for u in all_users if not u.is_disabled()]),
        'disabled_users': len([u for u in all_users if u.is_disabled()]),
        'users_by_type': {
            'normal': len([u for u in all_users if u.type == UserType.NORMAL]),
            'special': len([u for u in all_users if u.type == UserType.SPECIAL]),
            'temporary': len([u for u in all_users if u.type == UserType.TEMPORARY])
        },
        'usage_stats': event_logger.get_usage_stats(),
        'event_counts': event_logger.get_event_counts()
    }
    
    return jsonify(stats)

@admin_bp.route('/events', methods=['GET'])
@require_admin_auth
def get_events():
    """Get system events with filtering"""
    token = request.args.get('token')
    event_type = request.args.get('type')
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    
    events = event_logger.get_events(
        token=token,
        event_type=event_type,
        limit=limit,
        offset=offset
    )
    
    return jsonify({
        'events': events,
        'pagination': {
            'limit': limit,
            'offset': offset,
            'has_more': len(events) == limit
        }
    })

@admin_bp.route('/bulk/refresh-quotas', methods=['POST'])
@require_admin_auth
def bulk_refresh_quotas():
    """Refresh quotas for all users"""
    data = request.get_json() or {}
    model_family = data.get('model_family', 'all')
    
    affected_users = 0
    
    for user in user_store.get_all_users():
        if user.is_disabled():
            continue
        
        # Reset token counts
        if model_family == 'all':
            for family in user.token_counts:
                user.token_counts[family] = type(user.token_counts[family])()
        elif model_family in user.token_counts:
            user.token_counts[model_family] = type(user.token_counts[model_family])()
        
        user_store.flush_queue.add(user.token)
        affected_users += 1
    
    # Log admin action
    event_logger.log_user_action(
        token='admin',
        action='bulk_refresh_quotas',
        details={
            'performed_by': 'admin',
            'model_family': model_family,
            'affected_users': affected_users
        }
    )
    
    return jsonify({
        'success': True,
        'affected_users': affected_users,
        'message': f'Refreshed quotas for {affected_users} users'
    })

# Public user stats endpoint (for logged-in users to view their own stats)
@admin_bp.route('/user/stats', methods=['GET'])
@require_auth
def get_my_user_stats():
    """Get comprehensive user statistics for the authenticated user"""
    # Get user token from auth data
    user_token = g.auth_data.get('token')
    user = user_store.get_user(user_token)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Calculate comprehensive stats
    total_tokens = sum(count.total for count in user.token_counts.values())
    total_input_tokens = sum(count.input for count in user.token_counts.values())
    total_output_tokens = sum(count.output for count in user.token_counts.values())
    total_cost = sum(count.cost_usd for count in user.token_counts.values())
    
    # Model usage stats
    model_stats = {}
    for family, count in user.token_counts.items():
        model_stats[family] = {
            'requests': count.requests,
            'input_tokens': count.input,
            'output_tokens': count.output,
            'total_tokens': count.total,
            'cost_usd': round(count.cost_usd, 4),
            'first_request': count.first_request.isoformat() if count.first_request else None,
            'last_request': count.last_request.isoformat() if count.last_request else None,
            'limit': user.get_token_limit(family),
            'usage_percentage': round((count.total / user.get_token_limit(family)) * 100, 2) if user.get_token_limit(family) > 0 else 0
        }
    
    # IP usage stats
    ip_stats = []
    for ip_usage in user.ip_usage:
        ip_stats.append({
            'ip_hash': ip_usage.ip,
            'requests': ip_usage.prompt_count,
            'total_tokens': ip_usage.total_tokens,
            'first_seen': ip_usage.first_seen.isoformat() if ip_usage.first_seen else None,
            'last_used': ip_usage.last_used.isoformat() if ip_usage.last_used else None,
            'models_used': ip_usage.models_used,
            'user_agent': ip_usage.user_agent,
            'is_suspicious': ip_usage.is_suspicious
        })
    
    # Calculate time-based stats
    days_since_created = (datetime.now() - user.created_at).days
    days_since_last_used = (datetime.now() - user.last_used).days if user.last_used else None
    
    # Get recent activity from event logger
    try:
        usage_stats = event_logger.get_usage_stats(user_token)
        recent_events = event_logger.get_events(token=user_token, limit=10)
    except:
        usage_stats = {}
        recent_events = []
    
    # Build response with cat-themed elements
    stats_response = {
        'user_info': {
            'token': user_token[:8] + '...',  # Partially masked for security
            'type': user.type.value,
            'nickname': user.nickname,
            'created_at': user.created_at.isoformat(),
            'last_used': user.last_used.isoformat() if user.last_used else None,
            'days_since_created': days_since_created,
            'days_since_last_used': days_since_last_used,
            'is_disabled': user.is_disabled(),
            'disabled_reason': user.disabled_reason
        },
        'usage_summary': {
            'total_requests': user.total_requests,
            'total_tokens': total_tokens,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_cost_usd': round(total_cost, 4),
            'average_tokens_per_request': round(total_tokens / user.total_requests, 2) if user.total_requests > 0 else 0,
            'ip_count': len(user.ip),
            'rate_limit_hits': user.rate_limit_hits,
            'suspicious_activity_count': user.suspicious_activity_count
        },
        'model_usage': model_stats,
        'ip_usage': ip_stats,
        'preferred_models': user.preferred_models,
        'cat_stats': {
            'mood': user.mood,
            'favorite_treat': user.favorite_treat,
            'paw_prints': user.paw_prints,
            'happiness_level': min(100, max(0, 100 - user.suspicious_activity_count * 10 + user.paw_prints * 2))
        },
        'limits': {
            'token_limits': user.token_limits,
            'prompt_limits': user.prompt_limits,
            'max_ips': user.max_ips,
            'remaining_prompts': user.get_remaining_prompts()
        },
        'recent_activity': {
            'usage_stats': usage_stats,
            'recent_events': recent_events[:5]  # Limit to last 5 events for security
        }
    }
    
    return jsonify(stats_response)