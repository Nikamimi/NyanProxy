from flask import Blueprint, request, jsonify, g
from typing import Dict, Any, List
from datetime import datetime

from ..middleware.auth import require_admin_auth, require_auth
from ..services.user_store import user_store, UserType
from ..services.firebase_logger import event_logger, structured_logger

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
        structured_logger.log_user_action(
            user_token=token,
            action='user_created',
            details={
                'created_by': 'admin',
                'type': user_type.value,
                'nickname': nickname
            },
            admin_user='admin'
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
    structured_logger.log_user_action(
        user_token=token,
        action='user_updated',
        details={
            'updated_by': 'admin',
            'changes': data
        },
        admin_user='admin'
    )
    
    return jsonify({
        'success': True,
        'message': 'User updated successfully'
    })

def require_admin_auth_or_session(f):
    """Decorator to require admin authentication via API key OR web session"""
    from functools import wraps
    from flask import session, jsonify, request
    from ..config.auth import auth_config
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for web session first
        if session.get('admin_authenticated'):
            return f(*args, **kwargs)
        
        # Check for admin key in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            admin_key = auth_header[7:]  # Remove "Bearer " prefix
            if admin_key == auth_config.admin_key:
                return f(*args, **kwargs)
        
        return jsonify({"error": "Admin authentication required"}), 401
    
    return decorated_function

@admin_bp.route('/users/<token>/disable', methods=['POST'])
@require_admin_auth_or_session
def disable_user(token: str):
    """Disable/ban a user"""
    data = request.get_json() or {}
    reason = data.get('reason', 'Disabled by admin')
    
    print(f"🐱 DEBUG disable_user API: Disabling user {token[:8]} with reason '{reason}'")
    
    # Check user state before disabling
    user_before = user_store.get_user(token)
    if not user_before:
        print(f"🚫 DEBUG disable_user API: User {token[:8]} not found")
        return jsonify({'error': 'User not found'}), 404
    
    print(f"🐱 DEBUG disable_user API: User {token[:8]} before - disabled_at: {user_before.disabled_at}")
    
    success = user_store.disable_user(token, reason)
    
    if not success:
        print(f"🚫 DEBUG disable_user API: Failed to disable user {token[:8]}")
        return jsonify({'error': 'User not found'}), 404
    
    # Check user state after disabling
    user_after = user_store.get_user(token)
    print(f"🔍 ADMIN DISABLE: After disable_user call:")
    print(f"🔍 ADMIN DISABLE: User {token[:8]} disabled_at: {user_after.disabled_at}")
    print(f"🔍 ADMIN DISABLE: User {token[:8]} disabled_reason: {user_after.disabled_reason}")
    print(f"🔍 ADMIN DISABLE: User {token[:8]} status: {getattr(user_after, 'status', 'not set')}")
    print(f"🔍 ADMIN DISABLE: User {token[:8]} is_disabled(): {user_after.is_disabled()}")
    print(f"🔍 ADMIN DISABLE: User {token[:8]} to_dict status: {user_after.to_dict().get('status')}")
    print(f"🐱 DEBUG disable_user API: User {token[:8]} after - disabled_at: {user_after.disabled_at}, reason: {user_after.disabled_reason}")
    
    # Log admin action
    structured_logger.log_user_action(
        user_token=token,
        action='user_disabled',
        details={
            'disabled_by': 'admin',
            'reason': reason
        },
        admin_user='admin'
    )
    
    return jsonify({
        'success': True,
        'message': 'User disabled successfully',
        'user_state': {
            'disabled_at': user_after.disabled_at.isoformat() if user_after.disabled_at else None,
            'disabled_reason': user_after.disabled_reason,
            'status': getattr(user_after, 'status', 'disabled').value if hasattr(getattr(user_after, 'status', None), 'value') else str(getattr(user_after, 'status', 'disabled')),
            'is_disabled': user_after.is_disabled()
        }
    })

@admin_bp.route('/users/<token>/enable', methods=['POST'])
@require_admin_auth_or_session
def enable_user(token: str):
    """Enable/unban a user"""
    print(f"🐱 DEBUG enable_user API: Enabling user {token[:8]}")
    
    # Check user state before enabling
    user_before = user_store.get_user(token)
    if not user_before:
        print(f"🚫 DEBUG enable_user API: User {token[:8]} not found")
        return jsonify({'error': 'User not found'}), 404
    
    print(f"🐱 DEBUG enable_user API: User {token[:8]} before - disabled_at: {user_before.disabled_at}, reason: {user_before.disabled_reason}")
    
    success = user_store.reactivate_user(token)
    
    if not success:
        print(f"🚫 DEBUG enable_user API: Failed to enable user {token[:8]}")
        return jsonify({'error': 'User not found'}), 404
    
    # Check user state after enabling
    user_after = user_store.get_user(token)
    print(f"🐱 DEBUG enable_user API: User {token[:8]} after - disabled_at: {user_after.disabled_at}, reason: {user_after.disabled_reason}")
    print(f"🐱 DEBUG enable_user API: User {token[:8]} is_disabled() returns: {user_after.is_disabled()}")
    
    # Verify the user is actually enabled
    if user_after.is_disabled():
        print(f"🚫 ERROR: User {token[:8]} should be enabled but is_disabled() still returns True!")
        return jsonify({'error': 'Failed to enable user - state not updated'}), 500
    
    # Log admin action
    structured_logger.log_user_action(
        user_token=token,
        action='user_enabled',
        details={
            'enabled_by': 'admin'
        },
        admin_user='admin'
    )
    
    return jsonify({
        'success': True,
        'message': 'User enabled successfully',
        'user_state': {
            'disabled_at': user_after.disabled_at.isoformat() if user_after.disabled_at else None,
            'disabled_reason': user_after.disabled_reason,
            'status': getattr(user_after, 'status', 'active').value if hasattr(getattr(user_after, 'status', None), 'value') else str(getattr(user_after, 'status', 'active')),
            'is_disabled': user_after.is_disabled()
        }
    })

@admin_bp.route('/users/<token>/rotate', methods=['POST'])
@require_admin_auth
def rotate_user_token(token: str):
    """Rotate user token"""
    new_token = user_store.rotate_user_token(token)
    
    if not new_token:
        return jsonify({'error': 'User not found'}), 404
    
    # Log admin action
    structured_logger.log_user_action(
        user_token=new_token,
        action='token_rotated',
        details={
            'rotated_by': 'admin',
            'old_token': token[:8] + '...'
        },
        admin_user='admin'
    )
    
    return jsonify({
        'success': True,
        'new_token': new_token,
        'message': 'Token rotated successfully'
    })

@admin_bp.route('/users/<token>', methods=['DELETE'])
@require_admin_auth_or_session
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
            structured_logger.log_user_action(
                user_token=token,
                action='user_disabled',
                details={
                    'disabled_by': 'admin',
                    'reason': 'Disabled before deletion by admin'
                },
                admin_user='admin'
            )
        
        # Now delete the user
        success = user_store.delete_user(token)
        
        if not success:
            return jsonify({'error': 'Failed to delete user'}), 500
        
        # Log admin action
        structured_logger.log_user_action(
            user_token=token,
            action='user_deleted',
            details={
                'deleted_by': 'admin'
            },
            admin_user='admin'
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

@admin_bp.route('/bulk/migrate-status', methods=['POST'])
@require_admin_auth
def bulk_migrate_status():
    """Migrate all users to add explicit status field"""
    from ..services.user_store import UserStatus
    migrated_count = 0
    
    print(f"🔄 BULK MIGRATION: Starting migration for {len(user_store.users)} users")
    
    for token, user in user_store.users.items():
        print(f"🔄 BULK MIGRATION: Checking user {token[:8]} - disabled_at: {user.disabled_at}, status: {getattr(user, 'status', 'not set')}")
        
        # Always ensure status field exists and is correct
        if not hasattr(user, 'status') or user.status is None:
            print(f"🔄 BULK MIGRATION: User {token[:8]} missing status field")
            if user.disabled_at:
                user.status = UserStatus.DISABLED
                print(f"🔄 BULK MIGRATION: Set user {token[:8]} status to DISABLED")
            else:
                user.status = UserStatus.ACTIVE
                print(f"🔄 BULK MIGRATION: Set user {token[:8]} status to ACTIVE")
            migrated_count += 1
        else:
            # Check if status is inconsistent with disabled_at and fix
            if user.disabled_at and user.status == UserStatus.ACTIVE:
                print(f"🔄 BULK MIGRATION: User {token[:8]} inconsistent - disabled but status is ACTIVE")
                user.status = UserStatus.DISABLED
                migrated_count += 1
            elif not user.disabled_at and user.status != UserStatus.ACTIVE:
                print(f"🔄 BULK MIGRATION: User {token[:8]} inconsistent - not disabled but status is {user.status}")
                user.status = UserStatus.ACTIVE
                migrated_count += 1
        
        # Force immediate flush to Firebase for this user
        print(f"🔄 BULK MIGRATION: Force flushing user {token[:8]} to Firebase")
        user_store._flush_to_firebase(token)
    
    print(f"🔄 BULK MIGRATION: Completed migration of {migrated_count} users")
    
    return jsonify({
        'success': True,
        'migrated_users': migrated_count,
        'total_users': len(user_store.users),
        'message': f'Migrated {migrated_count} out of {len(user_store.users)} users to explicit status field'
    })

@admin_bp.route('/debug/user-state/<token>', methods=['GET'])
@require_admin_auth
def debug_user_state(token: str):
    """Debug user state in memory vs Firebase"""
    print(f"🔍 DEBUG ENDPOINT: Checking user {token[:8]} state")
    
    # Get user from memory
    memory_user = user_store.get_user(token)
    if not memory_user:
        return jsonify({'error': 'User not found in memory'}), 404
    
    print(f"🔍 DEBUG ENDPOINT: Memory user {token[:8]} status: {getattr(memory_user, 'status', 'not set')}")
    print(f"🔍 DEBUG ENDPOINT: Memory user {token[:8]} disabled_at: {memory_user.disabled_at}")
    print(f"🔍 DEBUG ENDPOINT: Memory user {token[:8]} is_disabled(): {memory_user.is_disabled()}")
    
    # Get user from Firebase
    firebase_user_data = None
    if user_store.firebase_db:
        try:
            sanitized_token = user_store._sanitize_firebase_key(token)
            firebase_user_data = user_store.firebase_db.child('users').child(sanitized_token).get()
            print(f"🔍 DEBUG ENDPOINT: Firebase user {token[:8]} status: {firebase_user_data.get('status') if firebase_user_data else 'no data'}")
            print(f"🔍 DEBUG ENDPOINT: Firebase user {token[:8]} disabled_at: {firebase_user_data.get('disabled_at') if firebase_user_data else 'no data'}")
        except Exception as e:
            firebase_user_data = f"Error: {e}"
    
    return jsonify({
        'token': token[:8] + '...',
        'memory_user': {
            'disabled_at': memory_user.disabled_at.isoformat() if memory_user.disabled_at else None,
            'disabled_reason': memory_user.disabled_reason,
            'status': getattr(memory_user, 'status', 'not set').value if hasattr(getattr(memory_user, 'status', None), 'value') else str(getattr(memory_user, 'status', 'not set')),
            'hasattr_status': hasattr(memory_user, 'status'),
            'status_type': str(type(getattr(memory_user, 'status', None))),
            'is_disabled': memory_user.is_disabled(),
            'to_dict_status': memory_user.to_dict().get('status'),
            'to_dict_disabled_at': memory_user.to_dict().get('disabled_at'),
            'to_dict_disabled_reason': memory_user.to_dict().get('disabled_reason')
        },
        'firebase_user': firebase_user_data,
        'in_flush_queue': token in user_store.flush_queue
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
    structured_logger.log_user_action(
        user_token='admin',
        action='bulk_refresh_quotas',
        details={
            'performed_by': 'admin',
            'model_family': model_family,
            'affected_users': affected_users
        },
        admin_user='admin'
    )
    
    return jsonify({
        'success': True,
        'affected_users': affected_users,
        'message': f'Refreshed quotas for {affected_users} users'
    })

@admin_bp.route('/bulk/reactivate-by-rules', methods=['POST'])
@require_admin_auth
def bulk_reactivate_by_current_rules():
    """Reactivate users who no longer violate current anti-abuse rules"""
    reactivated_count = 0
    eligible_users = []
    
    # Get current anti-abuse config
    current_max_ips = auth_config.max_ips_per_user
    
    for user in user_store.get_all_users():
        if not user.is_disabled():
            continue
            
        # Check if user should be reactivated based on current rules
        should_reactivate = False
        reason = ""
        
        if user.disabled_reason == "IP address limit exceeded":
            if len(user.ip) <= current_max_ips:
                should_reactivate = True
                reason = f"IP count ({len(user.ip)}) now within limit ({current_max_ips})"
        
        elif user.disabled_reason == "Prompt limit exceeded":
            # Check against current prompt limits for temporary users
            if hasattr(user, 'prompt_count') and not user.is_prompt_limit_exceeded():
                should_reactivate = True
                reason = "Prompt count now within current limits"
        
        if should_reactivate:
            eligible_users.append({
                'token': user.token[:8] + '...',
                'disable_reason': user.disabled_reason,
                'reactivation_reason': reason,
                'ip_count': len(user.ip) if hasattr(user, 'ip') else 0
            })
            
            # Reactivate the user
            user_store.enable_user(user.token, f"Auto-reactivated: {reason}")
            reactivated_count += 1
    
    # Log admin action
    structured_logger.log_user_action(
        user_token='admin',
        action='bulk_reactivate_by_rules',
        details={
            'performed_by': 'admin',
            'reactivated_count': reactivated_count,
            'current_max_ips': current_max_ips,
            'eligible_users': len(eligible_users)
        },
        admin_user='admin'
    )
    
    return jsonify({
        'success': True,
        'reactivated_count': reactivated_count,
        'eligible_users': eligible_users,
        'current_rules': {
            'max_ips_per_user': current_max_ips
        },
        'message': f'Reactivated {reactivated_count} users based on current anti-abuse rules'
    })

@admin_bp.route('/bulk/reactivate-selected', methods=['POST'])
@require_admin_auth
def bulk_reactivate_selected():
    """Reactivate specific selected users"""
    data = request.get_json() or {}
    user_tokens = data.get('tokens', [])
    reactivation_reason = data.get('reason', 'Bulk admin reactivation')
    
    reactivated_count = 0
    failed_tokens = []
    
    for token in user_tokens:
        try:
            if user_store.enable_user(token, reactivation_reason):
                reactivated_count += 1
            else:
                failed_tokens.append(token[:8] + '...')
        except Exception as e:
            failed_tokens.append(token[:8] + '...')
            print(f"Failed to reactivate user {token[:8]}: {e}")
    
    # Log admin action
    structured_logger.log_user_action(
        user_token='admin',
        action='bulk_reactivate_selected',
        details={
            'performed_by': 'admin',
            'reactivated_count': reactivated_count,
            'total_requested': len(user_tokens),
            'failed_count': len(failed_tokens),
            'reason': reactivation_reason
        },
        admin_user='admin'
    )
    
    return jsonify({
        'success': True,
        'reactivated_count': reactivated_count,
        'total_requested': len(user_tokens),
        'failed_tokens': failed_tokens,
        'message': f'Reactivated {reactivated_count}/{len(user_tokens)} users'
    })

@admin_bp.route('/bulk/preview-reactivation', methods=['GET'])
@require_admin_auth
def preview_bulk_reactivation():
    """Preview which users would be reactivated by current rules"""
    eligible_users = []
    
    # Get current anti-abuse config
    current_max_ips = auth_config.max_ips_per_user
    
    for user in user_store.get_all_users():
        if not user.is_disabled():
            continue
            
        should_reactivate = False
        reason = ""
        
        if user.disabled_reason == "IP address limit exceeded":
            if len(user.ip) <= current_max_ips:
                should_reactivate = True
                reason = f"IP count ({len(user.ip)}) now within limit ({current_max_ips})"
        
        elif user.disabled_reason == "Prompt limit exceeded":
            if hasattr(user, 'prompt_count') and not user.is_prompt_limit_exceeded():
                should_reactivate = True
                reason = "Prompt count now within current limits"
        
        if should_reactivate:
            eligible_users.append({
                'token': user.token[:8] + '...',
                'disable_reason': user.disabled_reason,
                'disabled_at': user.disabled_at.isoformat() if user.disabled_at else None,
                'reactivation_reason': reason,
                'ip_count': len(user.ip) if hasattr(user, 'ip') else 0,
                'type': user.type.value if hasattr(user, 'type') else 'unknown'
            })
    
    return jsonify({
        'success': True,
        'eligible_count': len(eligible_users),
        'eligible_users': eligible_users,
        'current_rules': {
            'max_ips_per_user': current_max_ips
        }
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
            'happiness_level': user.happiness,
            'suspicious_events': user.suspicious_events,
            'token_violations': user.token_violations,
            'rate_limit_violations': user.rate_limit_violations
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

@admin_bp.route('/model-usage-stats', methods=['GET'])
def get_model_usage_stats():
    """Get all active/whitelisted models with their usage statistics for the dashboard"""
    try:
        from ..services.model_families import model_manager, AIProvider
        
        # Get global usage stats for models that have been used
        global_stats = model_manager.get_usage_stats()
        
        # Organize stats by provider, including all whitelisted models
        provider_stats = {}
        
        # Iterate through all providers and their whitelisted models
        for provider in AIProvider:
            provider_key = provider.value
            whitelisted_models = model_manager.get_whitelisted_models(provider)
            
            if not whitelisted_models:
                continue
                
            provider_stats[provider_key] = {}
            
            for model_info in whitelisted_models:
                model_id = model_info.model_id
                sanitized_key = model_manager.sanitize_key(model_id)
                
                # Get usage stats if available, otherwise use zeros
                model_stats = global_stats.get(model_id, {})
                
                provider_stats[provider_key][sanitized_key] = {
                    'model_name': model_info.display_name,
                    'model_id': model_id,
                    'description': model_info.description,
                    'cat_personality': model_info.cat_personality,
                    'is_premium': model_info.is_premium,
                    'total_requests': model_stats.get('total_requests', 0),
                    'total_input_tokens': model_stats.get('total_input_tokens', 0),
                    'total_output_tokens': model_stats.get('total_output_tokens', 0),
                    'total_cost': model_stats.get('total_cost', 0.0),
                    'success_rate': model_stats.get('success_rate', 100.0),
                    'last_used': model_stats.get('last_used'),
                    'first_used': model_stats.get('first_used'),
                    'has_usage': model_id in global_stats
                }
        
        return jsonify(provider_stats)
        
    except Exception as e:
        # Log the error but return empty stats to avoid breaking the dashboard
        print(f"Error fetching model usage stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({})

@admin_bp.route('/save-model-stats', methods=['POST'])
@require_admin_auth
def save_model_stats():
    """Manually save model usage statistics to Firebase"""
    try:
        from ..services.model_families import model_manager
        
        # Force save current stats
        model_manager.save_usage_stats_sync()
        
        return jsonify({
            'success': True,
            'message': 'Model usage statistics saved successfully'
        })
        
    except Exception as e:
        print(f"Error saving model usage stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Error saving stats: {str(e)}'
        }), 500

@admin_bp.route('/debug-model-stats', methods=['GET'])
@require_admin_auth  
def debug_model_stats():
    """Debug endpoint to see current model stats in memory and Firebase"""
    try:
        from ..services.model_families import model_manager
        
        # Get current in-memory stats
        memory_stats = {}
        with model_manager.lock:
            for model_id, stats in model_manager.global_usage_stats.items():
                memory_stats[model_id] = {
                    'total_requests': stats.total_requests,
                    'total_input_tokens': stats.total_input_tokens, 
                    'total_output_tokens': stats.total_output_tokens,
                    'total_cost': stats.total_cost,
                    'last_used': stats.last_used.isoformat() if stats.last_used else None
                }
        
        # Get Firebase data
        firebase_stats = {}
        if model_manager.firebase_db:
            try:
                stats_ref = model_manager.firebase_db.child('global_usage_stats')
                firebase_data = stats_ref.get()
                if firebase_data:
                    for sanitized_key, stats_data in firebase_data.items():
                        model_id = model_manager.unsanitize_key(sanitized_key)
                        firebase_stats[model_id] = stats_data
            except Exception as e:
                firebase_stats = {'error': str(e)}
        
        return jsonify({
            'memory_stats': memory_stats,
            'firebase_stats': firebase_stats,
            'save_counter': model_manager._save_counter,
            'firebase_connected': model_manager.firebase_db is not None
        })
        
    except Exception as e:
        print(f"Error in debug endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/global-totals', methods=['GET'])
def get_global_totals():
    """Get global totals for System Metrics (public endpoint - no auth required)"""
    try:
        from ..services.model_families import model_manager
        
        # Get global totals from model manager
        global_totals = model_manager.get_global_totals()
        
        # Calculate total tokens (input + output)
        total_tokens = (global_totals.get('total_input_tokens', 0) + 
                       global_totals.get('total_output_tokens', 0))
        
        return jsonify({
            'total_requests': global_totals.get('total_requests', 0),
            'total_tokens': total_tokens,
            'total_input_tokens': global_totals.get('total_input_tokens', 0),
            'total_output_tokens': global_totals.get('total_output_tokens', 0),
            'total_cost': global_totals.get('total_cost', 0.0),
            'total_models_used': global_totals.get('total_models_used', 0),
            'first_request': global_totals.get('first_request'),
            'last_request': global_totals.get('last_request')
        })
        
    except Exception as e:
        print(f"Error fetching global totals: {e}")
        # Return zeros instead of error to avoid breaking dashboard
        return jsonify({
            'total_requests': 0,
            'total_tokens': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_cost': 0.0,
            'total_models_used': 0,
            'first_request': None,
            'last_request': None
        })

@admin_bp.route('/session/save', methods=['POST'])
def save_user_session():
    """Save user session based on IP address"""
    try:
        data = request.get_json()
        user_token = data.get('user_token')
        
        if not user_token:
            return jsonify({'success': False, 'message': 'No user token provided'}), 400
        
        # Get client IP address
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        if ',' in client_ip:
            client_ip = client_ip.split(',')[0].strip()
        
        # Validate the user token exists
        from ..services.user_store import user_store
        user = user_store.get_user_by_token(user_token)
        if not user:
            return jsonify({'success': False, 'message': 'Invalid user token'}), 401
        
        # Store session in a simple in-memory dict (could be moved to Firebase for persistence)
        if not hasattr(save_user_session, 'sessions'):
            save_user_session.sessions = {}
        
        save_user_session.sessions[client_ip] = {
            'user_token': user_token,
            'created_at': datetime.now().isoformat(),
            'user_info': {
                'token': user.token,
                'type': user.type.value,
                'nickname': user.nickname
            }
        }
        
        return jsonify({
            'success': True,
            'message': 'Session saved',
            'ip': client_ip
        })
        
    except Exception as e:
        print(f"Error saving user session: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@admin_bp.route('/session/load', methods=['GET'])
def load_user_session():
    """Load user session based on IP address"""
    try:
        # Get client IP address
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        if ',' in client_ip:
            client_ip = client_ip.split(',')[0].strip()
        
        # Check if session exists
        if not hasattr(save_user_session, 'sessions'):
            return jsonify({'has_session': False})
        
        session_data = save_user_session.sessions.get(client_ip)
        if not session_data:
            return jsonify({'has_session': False})
        
        # Validate the stored token still exists
        from ..services.user_store import user_store
        user = user_store.get_user_by_token(session_data['user_token'])
        if not user:
            # Remove invalid session
            del save_user_session.sessions[client_ip]
            return jsonify({'has_session': False})
        
        return jsonify({
            'has_session': True,
            'user_token': session_data['user_token'],
            'user_info': session_data['user_info'],
            'created_at': session_data['created_at']
        })
        
    except Exception as e:
        print(f"Error loading user session: {e}")
        return jsonify({'has_session': False})

@admin_bp.route('/session/clear', methods=['POST'])
def clear_user_session():
    """Clear user session based on IP address"""
    try:
        # Get client IP address
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        if ',' in client_ip:
            client_ip = client_ip.split(',')[0].strip()
        
        # Remove session if it exists
        if hasattr(save_user_session, 'sessions') and client_ip in save_user_session.sessions:
            del save_user_session.sessions[client_ip]
        
        return jsonify({'success': True, 'message': 'Session cleared'})
        
    except Exception as e:
        print(f"Error clearing user session: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500