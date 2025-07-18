from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session, flash
from typing import Dict, Any, List
from datetime import datetime
import os

from ..middleware.auth import require_admin_auth, require_admin_session
from ..services.user_store import user_store, UserType
from ..services.firebase_logger import event_logger, structured_logger
from ..config.auth import auth_config

admin_web_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_web_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Admin login page"""
    if request.method == 'POST':
        admin_key = request.form.get('admin_key')
        
        if admin_key == auth_config.admin_key:
            session['admin_authenticated'] = True
            session.permanent = True
            flash('Successfully logged in', 'success')
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid admin key', 'error')
    
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/login.html', config=dashboard_config)

@admin_web_bp.route('/logout')
def logout():
    """Admin logout"""
    session.pop('admin_authenticated', None)
    flash('Successfully logged out', 'info')
    return redirect(url_for('admin.login'))

@admin_web_bp.route('/')
@require_admin_session
def dashboard():
    """Admin dashboard"""
    # Get system statistics
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
        'usage_stats': structured_logger.get_analytics(),
        'event_counts': {}  # Firebase logger doesn't have this method
    }
    
    # Get recent events (Firebase logger doesn't have this method yet)
    recent_events = []  # TODO: Implement get_events method in FirebaseEventLogger
    
    # Get first 10 users for dashboard display
    recent_users = all_users[:10]
    users_data = []
    for user in recent_users:
        total_tokens = sum(count.total for count in user.token_counts.values())
        total_requests = sum(usage.prompt_count for usage in user.ip_usage)
        
        user_data = {
            'user': user,
            'stats': {
                'total_tokens': total_tokens,
                'total_requests': total_requests,
                'ip_count': len(user.ip),
                'days_since_created': (datetime.now() - user.created_at).days
            }
        }
        users_data.append(user_data)
    
    dashboard_config = {
        'title': f"{os.getenv('BRAND_EMOJI', '🐱')} {os.getenv('DASHBOARD_TITLE', 'NyanProxy Admin')}",
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/dashboard.html',
                         config=dashboard_config,
                         stats=stats,
                         recent_events=recent_events,
                         users_data=users_data,
                         auth_config=auth_config,
                         firebase_connected=hasattr(user_store, 'firebase_db') and user_store.firebase_db is not None)

@admin_web_bp.route('/users')
@require_admin_session
def list_users():
    """List all users with pagination and filtering"""
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 25))
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
    
    # Add usage statistics to user objects
    users_data = []
    for user in paginated_users:
        # Add usage statistics
        total_tokens = sum(count.total for count in user.token_counts.values())
        total_requests = sum(usage.prompt_count for usage in user.ip_usage)
        
        # Create a user data object that preserves datetime objects
        user_data = {
            'user': user,  # Keep the original user object with datetime fields
            'stats': {
                'total_tokens': total_tokens,
                'total_requests': total_requests,
                'ip_count': len(user.ip),
                'days_since_created': (datetime.now() - user.created_at).days
            }
        }
        
        users_data.append(user_data)
    
    pagination = {
        'page': page,
        'limit': limit,
        'total': len(filtered_users),
        'has_next': end < len(filtered_users)
    }
    
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/list_users.html',
                         config=dashboard_config,
                         users=users_data,
                         pagination=pagination)

@admin_web_bp.route('/users/create', methods=['GET', 'POST'])
@require_admin_session
def create_user():
    """Create a new user"""
    if request.method == 'POST':
        user_type = UserType(request.form.get('type', 'normal'))
        nickname = request.form.get('nickname') or None
        
        # Parse token limits
        token_limits = {}
        if request.form.get('openai_limit'):
            token_limits['openai'] = int(request.form.get('openai_limit'))
        if request.form.get('anthropic_limit'):
            token_limits['anthropic'] = int(request.form.get('anthropic_limit'))
        
        # Parse temporary user options
        prompt_limits = None
        max_ips = None
        if user_type == UserType.TEMPORARY:
            if request.form.get('prompt_limits'):
                prompt_limits = int(request.form.get('prompt_limits'))
            if request.form.get('max_ips'):
                max_ips = int(request.form.get('max_ips'))
        
        try:
            token = user_store.create_user(
                user_type=user_type,
                token_limits=token_limits,
                nickname=nickname,
                prompt_limits=prompt_limits,
                max_ips=max_ips
            )
            
            # Log admin action (both old and new loggers)
            event_logger.log_user_action(
                token=token,
                action='user_created',
                details={
                    'created_by': 'admin',
                    'type': user_type.value,
                    'nickname': nickname,
                    'prompt_limits': prompt_limits,
                    'max_ips': max_ips
                }
            )
            
            # Also log with structured logger
            structured_logger.log_user_action(
                user_token=token,
                action='user_created',
                details={
                    'created_by': 'admin',
                    'type': user_type.value,
                    'nickname': nickname,
                    'token_limits': token_limits,
                    'prompt_limits': prompt_limits,
                    'max_ips': max_ips
                },
                admin_user='admin'
            )
            
            flash(f'User created successfully! Token: {token}', 'success')
            return redirect(url_for('admin.create_user') + '?created=true')
            
        except Exception as e:
            flash(f'Error creating user: {str(e)}', 'error')
    
    # Get recent users for display
    all_users = user_store.get_all_users()
    recent_users = sorted(all_users, key=lambda u: u.created_at, reverse=True)[:5]
    
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/create_user.html', 
                         config=dashboard_config,
                         recent_users=recent_users,
                         new_user_created=request.args.get('created') == 'true')

@admin_web_bp.route('/users/<token>')
@require_admin_session
def view_user(token: str):
    """View detailed user information"""
    user = user_store.get_user(token)
    
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin.list_users'))
    
    # Add detailed statistics from both loggers
    usage_stats = event_logger.get_user_stats(token)
    
    # Get enhanced stats from structured logger
    enhanced_stats = structured_logger.get_user_analytics(token)
    
    # Combine stats for better reporting
    if enhanced_stats:
        usage_stats.update({
            'total_cost': enhanced_stats.get('total_cost', 0),
            'avg_response_time': enhanced_stats.get('avg_response_time', 0),
            'success_rate': enhanced_stats.get('success_rate', 100),
            'models_used': enhanced_stats.get('models_used', []),
            'hourly_distribution': enhanced_stats.get('hourly_distribution', {}),
            'daily_usage': enhanced_stats.get('daily_usage', {})
        })
    
    # Add recent events (Firebase logger doesn't have this method yet)
    recent_events = []  # TODO: Implement get_events method in FirebaseEventLogger
    
    # Create user data object that preserves datetime objects
    user_data = {
        'user': user,
        'usage_stats': usage_stats,
        'recent_events': recent_events
    }
    
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/view_user.html',
                         config=dashboard_config,
                         user_data=user_data)

@admin_web_bp.route('/users/<token>/edit', methods=['GET', 'POST'])
@require_admin_session
def edit_user(token: str):
    """Edit user properties"""
    user = user_store.get_user(token)
    
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin.list_users'))
    
    if request.method == 'POST':
        # Update nickname
        if 'nickname' in request.form:
            user.nickname = request.form['nickname']
        
        # Update token limits
        token_limits = {}
        if request.form.get('openai_limit'):
            token_limits['openai'] = int(request.form.get('openai_limit'))
        if request.form.get('anthropic_limit'):
            token_limits['anthropic'] = int(request.form.get('anthropic_limit'))
        
        if token_limits:
            user.token_limits.update(token_limits)
        
        # Update user type
        if 'type' in request.form:
            user.type = UserType(request.form['type'])
        
        # Add to flush queue
        user_store.flush_queue.add(token)
        
        # Log admin action
        event_logger.log_user_action(
            token=token,
            action='user_updated',
            details={
                'updated_by': 'admin',
                'changes': dict(request.form)
            }
        )
        
        flash('User updated successfully', 'success')
        return redirect(url_for('admin.view_user', token=token))
    
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/edit_user.html',
                         config=dashboard_config,
                         user=user)

@admin_web_bp.route('/key-manager')
@require_admin_session
def key_manager():
    """API key management interface"""
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/key_manager.html', config=dashboard_config)

@admin_web_bp.route('/anti-abuse')
@require_admin_session
def anti_abuse():
    """Anti-abuse settings interface"""
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/anti_abuse.html', config=dashboard_config)

@admin_web_bp.route('/stats')
@require_admin_session
def stats():
    """Statistics and analytics interface"""
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/stats.html', config=dashboard_config)

@admin_web_bp.route('/bulk-operations')
@require_admin_session
def bulk_operations():
    """Bulk operations interface"""
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    return render_template('admin/bulk_operations.html', config=dashboard_config)

# Add AJAX routes for frontend user management operations
@admin_web_bp.route('/users/<token>', methods=['DELETE'])
@require_admin_session
def delete_user_web(token: str):
    """Delete a user via AJAX (automatically disables first if needed)"""
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

@admin_web_bp.route('/users/<token>/disable', methods=['POST'])
@require_admin_session
def disable_user_web(token: str):
    """Disable a user via AJAX"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json() or {}
            reason = data.get('reason', 'Disabled by admin')
        else:
            reason = request.form.get('reason', 'Disabled by admin')
        
        # Attempting to disable user
        
        # Check if user exists first
        user = user_store.get_user(token)
        if not user:
            # User not found
            return jsonify({'error': 'User not found'}), 404
        
        # User found, checking status
        
        success = user_store.disable_user(token, reason)
        
        # Disable operation completed
        
        if not success:
            return jsonify({'error': 'Failed to disable user'}), 500
        
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
        
    except Exception as e:
        # Exception occurred during disable operation
        return jsonify({
            'success': False,
            'error': f'Failed to disable user: {str(e)}'
        }), 500

@admin_web_bp.route('/users/<token>/enable', methods=['POST'])
@require_admin_session
def enable_user_web(token: str):
    """Enable a user via AJAX"""
    try:
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
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to enable user: {str(e)}'
        }), 500

# Include the existing API routes for AJAX calls
from .admin import admin_bp as admin_api_bp