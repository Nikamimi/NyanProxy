from flask import Blueprint, request, jsonify
from typing import Dict, Any, List
from datetime import datetime

from ..middleware.auth import require_admin_auth
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