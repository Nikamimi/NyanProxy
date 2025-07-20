from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session, flash
from typing import Dict, Any, List
from datetime import datetime
import os

from ..middleware.auth import require_admin_auth, require_admin_session
from ..services.user_store import user_store, UserType
from ..services.firebase_logger import event_logger, structured_logger
from ..config.auth import auth_config

def bulk_reactivate_by_current_rules_internal():
    """Internal function to reactivate users who no longer violate current anti-abuse rules"""
    reactivated_count = 0
    
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
                reason = f"Auto-reactivated: IP count ({len(user.ip)}) now within limit ({current_max_ips})"
        
        elif user.disabled_reason == "Prompt limit exceeded":
            # Check against current prompt limits for temporary users
            if hasattr(user, 'prompt_count') and not user.is_prompt_limit_exceeded():
                should_reactivate = True
                reason = "Auto-reactivated: Prompt count now within current limits"
        
        if should_reactivate:
            # Reactivate the user
            user_store.enable_user(user.token, reason)
            reactivated_count += 1
    
    return reactivated_count

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
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 25))
        user_type = request.args.get('type')
        search = request.args.get('search', '').lower()
        
        print(f"🐱 DEBUG list_users: page={page}, limit={limit}, type={user_type}, search='{search}'")
        
        # Validate pagination parameters
        if page < 1:
            page = 1
        if limit < 1 or limit > 1000:
            limit = 25
        
        all_users = user_store.get_all_users()
        print(f"🐱 DEBUG: Found {len(all_users)} total users")
        
        # Apply filters with error handling
        filtered_users = []
        for i, user in enumerate(all_users):
            try:
                # Validate and fix user data integrity
                try:
                    if hasattr(user, 'validate_and_fix_data_integrity'):
                        if user.validate_and_fix_data_integrity():
                            print(f"🔧 DEBUG: Fixed data integrity issues for user {user.token[:8] if hasattr(user, 'token') else i}")
                            # Queue for Firebase update if fixes were made
                            with user_store.flush_queue_lock:
                                user_store.flush_queue.add(user.token)
                except Exception as e:
                    print(f"🚫 DEBUG: Error validating user {i}: {e}")
                
                # Ensure user has required attributes
                if not hasattr(user, 'type') or not hasattr(user, 'token'):
                    print(f"🚫 DEBUG: User {i} missing required attributes, skipping")
                    continue
                
                # Type filter
                if user_type and user.type.value != user_type:
                    continue
                
                # Search filter with safe string operations
                if search:
                    search_matches = False
                    try:
                        if search in user.token.lower():
                            search_matches = True
                        elif user.nickname and search in user.nickname.lower():
                            search_matches = True
                        elif user.disabled_reason and search in user.disabled_reason.lower():
                            search_matches = True
                    except Exception as e:
                        print(f"🚫 DEBUG: Error searching user {user.token[:8]}: {e}")
                        continue
                    
                    if search_matches:
                        filtered_users.append(user)
                else:
                    filtered_users.append(user)
                    
            except Exception as e:
                print(f"🚫 DEBUG: Error processing user {i}: {e}")
                continue
        
        print(f"🐱 DEBUG: Filtered to {len(filtered_users)} users")
        
        # Safe pagination
        start = (page - 1) * limit
        end = start + limit
        paginated_users = filtered_users[start:end]
        
        print(f"🐱 DEBUG: Paginated slice [{start}:{end}] = {len(paginated_users)} users")
        
        # Add usage statistics with error handling
        users_data = []
        for i, user in enumerate(paginated_users):
            try:
                # Safe statistics calculation
                total_tokens = 0
                total_requests = 0
                
                try:
                    if hasattr(user, 'token_counts') and user.token_counts:
                        total_tokens = sum(count.total for count in user.token_counts.values())
                except Exception as e:
                    print(f"🚫 DEBUG: Error calculating tokens for user {user.token[:8]}: {e}")
                
                try:
                    if hasattr(user, 'ip_usage') and user.ip_usage:
                        total_requests = sum(usage.prompt_count for usage in user.ip_usage)
                    elif hasattr(user, 'total_requests'):
                        total_requests = user.total_requests
                except Exception as e:
                    print(f"🚫 DEBUG: Error calculating requests for user {user.token[:8]}: {e}")
                
                # Safe datetime calculation
                days_since_created = 0
                try:
                    if hasattr(user, 'created_at') and user.created_at:
                        days_since_created = (datetime.now() - user.created_at).days
                except Exception as e:
                    print(f"🚫 DEBUG: Error calculating days for user {user.token[:8]}: {e}")
                
                # Ensure critical attributes exist for template rendering
                try:
                    if not hasattr(user, 'created_at') or user.created_at is None:
                        print(f"🚫 DEBUG: User {user.token[:8]} missing created_at, setting fallback")
                        user.created_at = datetime.now()
                    
                    if not hasattr(user, 'type') or user.type is None:
                        print(f"🚫 DEBUG: User {user.token[:8]} missing type, setting fallback")
                        from ..services.user_store import UserType
                        user.type = UserType.NORMAL
                    
                    if not hasattr(user, 'disabled_at'):
                        print(f"🚫 DEBUG: User {user.token[:8]} missing disabled_at, setting fallback")
                        user.disabled_at = None
                    
                    if not hasattr(user, 'nickname'):
                        print(f"🚫 DEBUG: User {user.token[:8]} missing nickname, setting fallback")
                        user.nickname = None
                    
                    if not hasattr(user, 'token') or not user.token:
                        print(f"🚫 DEBUG: User missing token, using provided token")
                        user.token = token
                    
                    # Ensure cat-themed attributes exist
                    if not hasattr(user, 'paw_prints'):
                        user.paw_prints = 0
                    if not hasattr(user, 'mood'):
                        user.mood = 'happy'
                    if not hasattr(user, 'total_cost'):
                        user.total_cost = 0.0
                    if not hasattr(user, 'mood_emoji'):
                        user.mood_emoji = '😸'
                        
                except Exception as e:
                    print(f"🚫 DEBUG: Error setting fallback attributes for user {user.token[:8] if hasattr(user, 'token') else 'unknown'}: {e}")
                
                user_data = {
                    'user': user,
                    'stats': {
                        'total_tokens': total_tokens,
                        'total_requests': total_requests,
                        'ip_count': len(user.ip) if hasattr(user, 'ip') and user.ip else 0,
                        'days_since_created': days_since_created
                    }
                }
                
                users_data.append(user_data)
                
            except Exception as e:
                print(f"🚫 DEBUG: Error creating user_data for user {i}: {e}")
                # Create minimal safe user data
                try:
                    user_data = {
                        'user': user,
                        'stats': {
                            'total_tokens': 0,
                            'total_requests': 0,
                            'ip_count': 0,
                            'days_since_created': 0
                        }
                    }
                    users_data.append(user_data)
                except Exception as e2:
                    print(f"🚫 DEBUG: Failed to create fallback user_data: {e2}")
                    continue
        
        total_filtered = len(filtered_users)
        total_pages = (total_filtered + limit - 1) // limit if total_filtered > 0 else 1
        
        pagination = {
            'page': page,
            'limit': limit,
            'total': total_filtered,
            'total_pages': total_pages,
            'has_next': end < total_filtered,
            'has_prev': page > 1
        }
        
        dashboard_config = {
            'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
            'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
        }
        
        print(f"🐱 DEBUG: Returning {len(users_data)} users for display")
        
        return render_template('admin/list_users.html',
                             config=dashboard_config,
                             users=users_data,
                             pagination=pagination)
                             
    except Exception as e:
        print(f"🚫 CRITICAL ERROR in list_users: {e}")
        import traceback
        traceback.print_exc()
        
        # Return safe fallback
        dashboard_config = {
            'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
            'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
        }
        
        flash(f'Error loading users: {str(e)}', 'error')
        
        return render_template('admin/list_users.html',
                             config=dashboard_config,
                             users=[],
                             pagination={'page': 1, 'limit': 25, 'total': 0, 'has_next': False})

@admin_web_bp.route('/users/create', methods=['GET', 'POST'])
@require_admin_session
def create_user():
    """Create a new user"""
    if request.method == 'POST':
        user_type = UserType(request.form.get('type', 'normal'))
        nickname = request.form.get('nickname') or None
        
        # Parse request limits
        request_limits = {}
        if request.form.get('openai_limit'):
            request_limits['openai'] = int(request.form.get('openai_limit'))
        if request.form.get('anthropic_limit'):
            request_limits['anthropic'] = int(request.form.get('anthropic_limit'))
        
        # Parse temporary user options
        temp_prompt_limits = None
        max_ips = None
        if user_type == UserType.TEMPORARY:
            if request.form.get('prompt_limits'):
                temp_prompt_limits = int(request.form.get('prompt_limits'))
            if request.form.get('max_ips'):
                max_ips = int(request.form.get('max_ips'))
        
        try:
            token = user_store.create_user(
                user_type=user_type,
                prompt_limits=request_limits,
                nickname=nickname,
                temp_prompt_limits=temp_prompt_limits,
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
    """View detailed user information with comprehensive error handling"""
    try:
        print(f"🐱 DEBUG view_user: Viewing user with token {token[:8]}...")
        
        user = user_store.get_user(token)
        
        if not user:
            print(f"🚫 DEBUG view_user: User {token[:8]} not found")
            flash('User not found', 'error')
            return redirect(url_for('admin.list_users'))
        
        print(f"🐱 DEBUG view_user: User {token[:8]} found, type={user.type.value if hasattr(user, 'type') else 'unknown'}")
        
        # Safe user stats collection with error handling
        usage_stats = {}
        enhanced_stats = {}
        
        try:
            print(f"🐱 DEBUG view_user: Getting event logger stats for {token[:8]}")
            usage_stats = event_logger.get_user_stats(token)
            print(f"🐱 DEBUG view_user: Event logger stats retrieved: {len(usage_stats) if usage_stats else 0} items")
        except Exception as e:
            print(f"🚫 DEBUG view_user: Error getting event logger stats for {token[:8]}: {e}")
            usage_stats = {}
        
        try:
            print(f"🐱 DEBUG view_user: Getting structured logger analytics for {token[:8]}")
            enhanced_stats = structured_logger.get_user_analytics(token)
            print(f"🐱 DEBUG view_user: Structured logger stats retrieved: {len(enhanced_stats) if enhanced_stats else 0} items")
        except Exception as e:
            print(f"🚫 DEBUG view_user: Error getting structured logger analytics for {token[:8]}: {e}")
            enhanced_stats = {}
        
        # Safe stats combination with error handling
        try:
            if enhanced_stats:
                safe_enhanced_stats = {
                    'total_cost': enhanced_stats.get('total_cost', 0),
                    'avg_response_time': enhanced_stats.get('avg_response_time', 0),
                    'success_rate': enhanced_stats.get('success_rate', 100),
                    'models_used': enhanced_stats.get('models_used', []),
                    'hourly_distribution': enhanced_stats.get('hourly_distribution', {}),
                    'daily_usage': enhanced_stats.get('daily_usage', {})
                }
                usage_stats.update(safe_enhanced_stats)
                print(f"🐱 DEBUG view_user: Stats combined successfully for {token[:8]}")
        except Exception as e:
            print(f"🚫 DEBUG view_user: Error combining stats for {token[:8]}: {e}")
        
        # Add recent events (Firebase logger doesn't have this method yet)
        recent_events = []  # TODO: Implement get_events method in FirebaseEventLogger
        
        # Safe user data creation with error handling
        try:
            # Validate user object has required attributes
            if not hasattr(user, 'token'):
                print(f"🚫 DEBUG view_user: User missing token attribute")
                user.token = token  # Fallback
            
            if not hasattr(user, 'type'):
                print(f"🚫 DEBUG view_user: User missing type attribute")
                from ..services.user_store import UserType
                user.type = UserType.NORMAL  # Fallback
            
            if not hasattr(user, 'created_at'):
                print(f"🚫 DEBUG view_user: User missing created_at attribute")
                user.created_at = datetime.now()  # Fallback
            
            user_data = {
                'user': user,
                'usage_stats': usage_stats,
                'recent_events': recent_events
            }
            
            print(f"🐱 DEBUG view_user: User data created successfully for {token[:8]}")
            
        except Exception as e:
            print(f"🚫 DEBUG view_user: Error creating user_data for {token[:8]}: {e}")
            # Create minimal safe user data
            user_data = {
                'user': user,
                'usage_stats': {},
                'recent_events': []
            }
        
        dashboard_config = {
            'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
            'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
        }
        
        print(f"🐱 DEBUG view_user: Rendering template for {token[:8]}")
        
        return render_template('admin/view_user.html',
                             config=dashboard_config,
                             user_data=user_data)
                             
    except Exception as e:
        print(f"🚫 CRITICAL ERROR in view_user for {token[:8]}: {e}")
        import traceback
        traceback.print_exc()
        
        # Return safe fallback
        dashboard_config = {
            'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
            'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
        }
        
        flash(f'Error loading user details: {str(e)}', 'error')
        
        # Try to create minimal user data for display
        try:
            user = user_store.get_user(token)
            if user:
                user_data = {
                    'user': user,
                    'usage_stats': {},
                    'recent_events': []
                }
                return render_template('admin/view_user.html',
                                     config=dashboard_config,
                                     user_data=user_data)
        except Exception as fallback_error:
            print(f"🚫 DEBUG view_user: Fallback also failed: {fallback_error}")
        
        # Ultimate fallback - redirect to user list
        return redirect(url_for('admin.list_users'))

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
        
        # Update prompt limits
        prompt_limits = {}
        if request.form.get('openai_limit'):
            prompt_limits['openai'] = int(request.form.get('openai_limit'))
        if request.form.get('anthropic_limit'):
            prompt_limits['anthropic'] = int(request.form.get('anthropic_limit'))
        
        if prompt_limits:
            user.prompt_limits.update(prompt_limits)
        
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

@admin_web_bp.route('/anti-abuse', methods=['GET', 'POST'])
@require_admin_session
def anti_abuse():
    """Anti-abuse settings interface with Firebase persistence"""
    dashboard_config = {
        'brand_name': os.getenv('BRAND_NAME', 'NyanProxy'),
        'brand_emoji': os.getenv('BRAND_EMOJI', '🐱'),
    }
    
    # Import config manager
    from ..services.config_manager import config_manager, AntiAbuseConfig
    
    if request.method == 'POST':
        try:
            # Get form data
            rate_limit_enabled = 'rate_limit_enabled' in request.form
            rate_limit_per_minute = int(request.form.get('rate_limit_per_minute', 60))
            max_ips_per_user = int(request.form.get('max_ips_per_user', 3))
            max_ips_auto_ban = 'max_ips_auto_ban' in request.form
            
            # Validate and clamp values
            rate_limit_per_minute = max(1, min(10000, rate_limit_per_minute))
            max_ips_per_user = max(1, min(1000, max_ips_per_user))
            
            # Create new config
            new_config = AntiAbuseConfig(
                rate_limit_enabled=rate_limit_enabled,
                rate_limit_per_minute=rate_limit_per_minute,
                max_ips_per_user=max_ips_per_user,
                max_ips_auto_ban=max_ips_auto_ban,
                updated_by=session.get('admin_user', 'admin')
            )
            
            # Save to Firebase
            if config_manager.save_anti_abuse_config(new_config):
                # Update runtime auth config
                from ..config.auth import auth_config
                auth_config.rate_limit_enabled = rate_limit_enabled
                auth_config.rate_limit_per_minute = rate_limit_per_minute
                auth_config.max_ips_per_user = max_ips_per_user
                auth_config.max_ips_auto_ban = max_ips_auto_ban
                
                # Update environment variables for immediate effect
                os.environ['RATE_LIMIT_ENABLED'] = str(rate_limit_enabled).lower()
                os.environ['RATE_LIMIT_PER_MINUTE'] = str(rate_limit_per_minute)
                os.environ['MAX_IPS_PER_USER'] = str(max_ips_per_user)
                os.environ['MAX_IPS_AUTO_BAN'] = str(max_ips_auto_ban).lower()
                
                # Check if auto-reactivation is requested
                auto_reactivate = request.form.get('auto_reactivate') == 'on'
                if auto_reactivate:
                    # Trigger bulk reactivation by current rules
                    reactivated_count = bulk_reactivate_by_current_rules_internal()
                    flash(f'🐱 Anti-hairball settings saved and {reactivated_count} users reactivated! Meow!', 'success')
                else:
                    flash('🐱 Anti-hairball settings saved to Firebase successfully! Meow!', 'success')
            else:
                flash('⚠️ Settings updated locally but Firebase save failed. Settings may not persist on restart.', 'warning')
            
        except ValueError as e:
            flash(f'❌ Invalid input: {str(e)}', 'error')
        except Exception as e:
            flash(f'❌ Error updating settings: {str(e)}', 'error')
        
        return redirect(url_for('admin.anti_abuse'))
    
    # Load current settings from Firebase
    try:
        # Load settings from Firebase first
        current_config = config_manager.load_anti_abuse_config()
        
        # Update runtime auth config with persisted values
        from ..config.auth import auth_config
        auth_config.rate_limit_enabled = current_config.rate_limit_enabled
        auth_config.rate_limit_per_minute = current_config.rate_limit_per_minute
        auth_config.max_ips_per_user = current_config.max_ips_per_user
        auth_config.max_ips_auto_ban = current_config.max_ips_auto_ban
        
        # Get current statistics
        from ..services.user_store import user_store
        all_users = user_store.get_all_users()
        
        # Calculate stats
        rate_limited_count = sum(user.rate_limit_hits for user in all_users)
        auto_banned_count = len([user for user in all_users if user.is_disabled() and user.disabled_reason and 'IP limit' in user.disabled_reason])
        blocked_ips = sum(len(user.ip) for user in all_users if user.is_disabled())
        
        dashboard_config.update({
            'current_settings': {
                'rate_limit_enabled': current_config.rate_limit_enabled,
                'rate_limit_per_minute': current_config.rate_limit_per_minute,
                'max_ips_per_user': current_config.max_ips_per_user,
                'max_ips_auto_ban': current_config.max_ips_auto_ban,
                'last_updated': current_config.last_updated,
                'updated_by': current_config.updated_by
            },
            'stats': {
                'rate_limited_requests': rate_limited_count,
                'auto_banned_users': auto_banned_count,
                'blocked_ips': blocked_ips
            }
        })
        
    except Exception as e:
        print(f"Error loading anti-abuse settings from Firebase: {e}")
        # Fallback to current auth config
        dashboard_config.update({
            'current_settings': {
                'rate_limit_enabled': auth_config.rate_limit_enabled,
                'rate_limit_per_minute': auth_config.rate_limit_per_minute,
                'max_ips_per_user': auth_config.max_ips_per_user,
                'max_ips_auto_ban': auth_config.max_ips_auto_ban,
                'last_updated': 'Unknown',
                'updated_by': 'system'
            },
            'stats': {
                'rate_limited_requests': 0,
                'auto_banned_users': 0,
                'blocked_ips': 0
            }
        })
    
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

# Note: User management AJAX operations are handled by the API routes in admin.py
# at /admin/api/users/<token>/... endpoints with session-based authentication

# Include the existing API routes for AJAX calls
from .admin import admin_bp as admin_api_bp