"""
🐱 Model Families Admin Interface
================================

Flask routes for managing model families through the admin interface.
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash
from typing import Dict, Any, List
import json

from ..middleware.auth import require_admin_session
from ..services.model_families import model_manager, AIProvider, ModelInfo
from ..services.firebase_logger import structured_logger

model_families_bp = Blueprint('model_families', __name__, url_prefix='/admin/model-families')

@model_families_bp.route('/')
@require_admin_session
def model_families_dashboard():
    """Model families management dashboard"""
    
    # Get all providers and their whitelisted models
    providers_data = {}
    total_models = 0
    total_whitelisted = 0
    
    for provider in AIProvider:
        all_models = model_manager.get_all_models(provider)
        whitelisted_models = model_manager.get_whitelisted_models(provider)
        
        # Define anchor default models that can never be deleted
        anchor_defaults = {
            "gpt-4o", "claude-3-5-sonnet-20241022"
        }
        
        # All default models (including ones that can be deleted)
        default_model_ids = {
            "gpt-4o", "claude-3-5-sonnet-20241022"
        }
        
        providers_data[provider.value] = {
            'name': provider.value.title(),
            'all_models': [
                {
                    'id': model.model_id,
                    'display_name': model.display_name,
                    'description': model.description,
                    'is_whitelisted': model in whitelisted_models,
                    'input_cost': model.input_cost_per_1m,
                    'output_cost': model.output_cost_per_1m,
                    'context_length': model.context_length,
                    'is_premium': model.is_premium,
                    'cat_emoji': model.get_cat_emoji(),
                    'cat_personality': model.cat_personality,
                    'max_input_tokens': model.max_input_tokens,
                    'max_output_tokens': model.max_output_tokens,
                    'is_default': model.model_id in default_model_ids,
                    'can_delete': model.model_id not in anchor_defaults,
                    'supports_streaming': model.supports_streaming,
                    'supports_function_calling': model.supports_function_calling
                }
                for model in all_models
            ],
            'whitelisted_count': len(whitelisted_models),
            'total_count': len(all_models)
        }
        
        total_models += len(all_models)
        total_whitelisted += len(whitelisted_models)
    
    # Get usage statistics
    global_usage = model_manager.get_usage_stats()
    cost_analysis = model_manager.get_cost_analysis()
    
    dashboard_config = {
        'brand_name': 'NyanProxy',
        'brand_emoji': '🐱',
    }
    
    return render_template('admin/model_families.html',
                         config=dashboard_config,
                         providers=providers_data,
                         total_models=total_models,
                         total_whitelisted=total_whitelisted,
                         usage_stats=global_usage,
                         cost_analysis=cost_analysis)

@model_families_bp.route('/provider/<provider_name>')
@require_admin_session  
def provider_details(provider_name: str):
    """Detailed view for a specific provider"""
    try:
        provider = AIProvider(provider_name.lower())
    except ValueError:
        flash(f'Unknown provider: {provider_name}', 'error')
        return redirect(url_for('model_families.model_families_dashboard'))
    
    all_models = model_manager.get_all_models(provider)
    whitelisted_models = model_manager.get_whitelisted_models(provider)
    
    models_data = []
    for model in all_models:
        usage_stats = model_manager.get_usage_stats(model_id=model.model_id)
        
        models_data.append({
            'info': model,
            'is_whitelisted': model in whitelisted_models,
            'usage_stats': usage_stats,
            'cost_per_request': usage_stats.get('total_cost', 0) / max(usage_stats.get('total_requests', 1), 1)
        })
    
    dashboard_config = {
        'brand_name': 'NyanProxy',
        'brand_emoji': '🐱',
    }
    
    return render_template('admin/provider_models.html',
                         config=dashboard_config,
                         provider=provider,
                         provider_name=provider_name.title(),
                         models=models_data)

@model_families_bp.route('/api/whitelist', methods=['POST'])
@require_admin_session
def update_whitelist():
    """Update model whitelist for a provider"""
    data = request.get_json()
    
    if not data or 'provider' not in data or 'models' not in data:
        return jsonify({'error': 'Invalid request data'}), 400
    
    try:
        provider = AIProvider(data['provider'].lower())
        model_ids = data['models']
        
        success = model_manager.set_provider_whitelist(provider, model_ids)
        
        if success:
            # Log admin action
            structured_logger.log_user_action(
                user_token='admin',
                action='model_whitelist_updated',
                details={
                    'provider': provider.value,
                    'models': model_ids,
                    'count': len(model_ids)
                },
                admin_user='admin'
            )
            
            return jsonify({
                'success': True,
                'message': f'Updated whitelist for {provider.value}',
                'whitelisted_count': len(model_ids)
            })
        else:
            return jsonify({'error': 'Some models were invalid'}), 400
            
    except ValueError as e:
        return jsonify({'error': f'Invalid provider: {data["provider"]}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_families_bp.route('/api/model/<model_id>/toggle', methods=['POST'])
@require_admin_session
def toggle_model_whitelist(model_id: str):
    """Toggle a single model's whitelist status"""
    
    # Find which provider this model belongs to
    model_info = model_manager.get_model_info(model_id)
    if not model_info:
        return jsonify({'error': 'Model not found'}), 404
    
    provider = model_info.provider
    is_whitelisted = model_manager.is_model_whitelisted(provider, model_id)
    
    if is_whitelisted:
        success = model_manager.remove_model_from_whitelist(provider, model_id)
        action = 'removed from'
    else:
        success = model_manager.add_model_to_whitelist(provider, model_id)
        action = 'added to'
    
    if success:
        # Log admin action
        structured_logger.log_user_action(
            user_token='admin',
            action='model_whitelist_toggled',
            details={
                'model_id': model_id,
                'provider': provider.value,
                'action': action,
                'display_name': model_info.display_name
            },
            admin_user='admin'
        )
        
        return jsonify({
            'success': True,
            'message': f'Model {action} whitelist',
            'is_whitelisted': not is_whitelisted
        })
    else:
        return jsonify({'error': 'Failed to update whitelist'}), 500

@model_families_bp.route('/api/usage-stats')
@require_admin_session
def get_usage_stats():
    """Get usage statistics for all models"""
    user_token = request.args.get('user_token')
    model_id = request.args.get('model_id')
    
    stats = model_manager.get_usage_stats(user_token=user_token, model_id=model_id)
    cost_analysis = model_manager.get_cost_analysis(user_token=user_token)
    
    return jsonify({
        'usage_stats': stats,
        'cost_analysis': cost_analysis
    })

@model_families_bp.route('/api/cost-analysis')
@require_admin_session
def get_cost_analysis():
    """Get detailed cost analysis"""
    user_token = request.args.get('user_token')
    
    analysis = model_manager.get_cost_analysis(user_token=user_token)
    
    return jsonify(analysis)

@model_families_bp.route('/usage')
@require_admin_session
def usage_analytics():
    """Usage analytics page"""
    
    # Get top models by usage
    global_stats = model_manager.get_usage_stats()
    cost_analysis = model_manager.get_cost_analysis()
    
    # Sort models by total cost
    top_models_by_cost = sorted(
        cost_analysis.get('by_model', {}).items(),
        key=lambda x: x[1]['cost'],
        reverse=True
    )[:10]
    
    # Sort models by requests
    top_models_by_requests = sorted(
        global_stats.items(),
        key=lambda x: x[1].get('total_requests', 0) if isinstance(x[1], dict) else 0,
        reverse=True
    )[:10]
    
    dashboard_config = {
        'brand_name': 'NyanProxy',
        'brand_emoji': '🐱',
    }
    
    return render_template('admin/model_usage.html',
                         config=dashboard_config,
                         cost_analysis=cost_analysis,
                         top_models_by_cost=top_models_by_cost,
                         top_models_by_requests=top_models_by_requests,
                         global_stats=global_stats)

@model_families_bp.route('/api/models', methods=['POST'])
@require_admin_session
def add_custom_model():
    """Add a new custom model"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate the model data
    errors = model_manager.validate_model_data(data)
    if errors:
        return jsonify({'error': 'Validation failed', 'field_errors': errors}), 400
    
    try:
        # Create ModelInfo object
        model_info = ModelInfo(
            model_id=data['model_id'],
            provider=AIProvider(data['provider'].lower()),
            display_name=data['display_name'],
            description=data['description'],
            input_cost_per_1m=float(data.get('input_cost_per_1m', 0)),
            output_cost_per_1m=float(data.get('output_cost_per_1m', 0)),
            context_length=int(data.get('context_length', 4096)),
            supports_streaming=bool(data.get('supports_streaming', True)),
            supports_function_calling=bool(data.get('supports_function_calling', False)),
            is_premium=bool(data.get('is_premium', False)),
            cat_personality=data.get('cat_personality', 'curious'),
            max_input_tokens=int(data['max_input_tokens']) if data.get('max_input_tokens') else None,
            max_output_tokens=int(data['max_output_tokens']) if data.get('max_output_tokens') else None
        )
        
        # Add the model
        auto_whitelist = data.get('auto_whitelist', True)
        print(f"🔥 API: Attempting to add model {model_info.model_id} to provider {model_info.provider.value}")
        
        try:
            success = model_manager.add_custom_model(model_info, auto_whitelist)
            print(f"🔥 API: add_custom_model returned: {success}")
            
            if success:
                # Log admin action asynchronously (non-blocking)
                import threading
                def background_log():
                    try:
                        structured_logger.log_user_action(
                            user_token='admin',
                            action='custom_model_added',
                            details={
                                'model_id': model_info.model_id,
                                'provider': model_info.provider.value,
                                'display_name': model_info.display_name,
                                'auto_whitelisted': auto_whitelist
                            },
                            admin_user='admin'
                        )
                    except Exception as log_error:
                        print(f"⚠️ Logging error (non-critical): {log_error}")
                
                log_thread = threading.Thread(target=background_log)
                log_thread.daemon = True
                log_thread.start()
                
                response = {
                    'success': True,
                    'message': f'Model {model_info.display_name} added successfully',
                    'model_id': model_info.model_id,
                    'whitelisted': auto_whitelist
                }
                print(f"✅ API: Returning success response: {response}")
                return jsonify(response)
            else:
                error_response = {'error': 'Failed to add model (already exists or save failed)'}
                print(f"❌ API: Returning error response: {error_response}")
                return jsonify(error_response), 400
        except Exception as add_error:
            error_response = {'error': f'Error adding model: {str(add_error)}'}
            print(f"❌ API: Exception in add_custom_model: {add_error}")
            import traceback
            traceback.print_exc()
            return jsonify(error_response), 500
            
    except ValueError as e:
        return jsonify({'error': f'Invalid data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@model_families_bp.route('/api/models/<model_id>', methods=['PUT'])
@require_admin_session
def update_custom_model(model_id: str):
    """Update an existing custom model"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # For updates, we don't need full validation - just update the provided fields
    # Skip validation for model updates since we're only updating specific fields
    
    try:
        # Convert numeric fields
        updates = {}
        for field in ['input_cost_per_1m', 'output_cost_per_1m', 'context_length']:
            if field in data:
                updates[field] = float(data[field]) if field != 'context_length' else int(data[field])
        
        # Convert token limit fields (can be None)
        for field in ['max_input_tokens', 'max_output_tokens']:
            if field in data:
                updates[field] = int(data[field]) if data[field] and data[field] != '' else None
        
        # Convert boolean fields
        for field in ['supports_streaming', 'supports_function_calling', 'is_premium']:
            if field in data:
                updates[field] = bool(data[field])
        
        # Copy string fields
        for field in ['display_name', 'description', 'cat_personality']:
            if field in data:
                updates[field] = data[field]
        
        success = model_manager.update_model_info(model_id, updates)
        
        if success:
            # Log admin action
            structured_logger.log_user_action(
                user_token='admin',
                action='custom_model_updated',
                details={
                    'model_id': model_id,
                    'updates': updates
                },
                admin_user='admin'
            )
            
            return jsonify({
                'success': True,
                'message': f'Model {model_id} updated successfully'
            })
        else:
            return jsonify({'error': 'Model not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@model_families_bp.route('/api/models/<model_id>', methods=['DELETE'])
@require_admin_session
def delete_custom_model(model_id: str):
    """Delete a model (custom or non-anchor default)"""
    try:
        # Define anchor defaults that cannot be deleted
        anchor_defaults = {
            "gpt-4o", "claude-3-5-sonnet-20241022"
        }
        
        # Prevent deletion of anchor defaults
        if model_id in anchor_defaults:
            return jsonify({'error': 'Cannot delete anchor default models'}), 403
        
        # Get model info for logging
        model_info = model_manager.get_model_info(model_id)
        if not model_info:
            return jsonify({'error': 'Model not found'}), 404
        
        success = model_manager.remove_custom_model(model_id)
        
        if success:
            # Log admin action
            structured_logger.log_user_action(
                user_token='admin',
                action='custom_model_deleted',
                details={
                    'model_id': model_id,
                    'display_name': model_info.display_name,
                    'provider': model_info.provider.value
                },
                admin_user='admin'
            )
            
            return jsonify({
                'success': True,
                'message': f'Model {model_info.display_name} deleted successfully'
            })
        else:
            return jsonify({'error': 'Failed to delete model'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@model_families_bp.route('/api/model-info/<model_id>')
@require_admin_session  
def get_model_info_api(model_id: str):
    """Get detailed info for a specific model"""
    try:
        model_info = model_manager.get_model_info(model_id)
        if not model_info:
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        
        model_dict = {
            'model_id': model_info.model_id,
            'provider': model_info.provider.value,
            'display_name': model_info.display_name,
            'description': model_info.description,
            'input_cost_per_1m': model_info.input_cost_per_1m,
            'output_cost_per_1m': model_info.output_cost_per_1m,
            'context_length': model_info.context_length,
            'supports_streaming': model_info.supports_streaming,
            'supports_function_calling': model_info.supports_function_calling,
            'is_premium': model_info.is_premium,
            'cat_personality': model_info.cat_personality,
            'max_input_tokens': model_info.max_input_tokens,
            'max_output_tokens': model_info.max_output_tokens
        }
        
        return jsonify({
            'success': True,
            'model': model_dict
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@model_families_bp.route('/api/models/custom')
@require_admin_session
def get_custom_models():
    """Get all custom models"""
    try:
        custom_models = model_manager.get_custom_models()
        
        models_data = []
        for model in custom_models:
            model_dict = {
                'model_id': model.model_id,
                'provider': model.provider.value,
                'display_name': model.display_name,
                'description': model.description,
                'input_cost_per_1m': model.input_cost_per_1m,
                'output_cost_per_1m': model.output_cost_per_1m,
                'context_length': model.context_length,
                'supports_streaming': model.supports_streaming,
                'supports_function_calling': model.supports_function_calling,
                'is_premium': model.is_premium,
                'cat_personality': model.cat_personality,
                'cat_emoji': model.get_cat_emoji(),
                'max_input_tokens': model.max_input_tokens,
                'max_output_tokens': model.max_output_tokens,
                'is_whitelisted': model_manager.is_model_whitelisted(model.provider, model.model_id)
            }
            models_data.append(model_dict)
        
        return jsonify({
            'success': True,
            'models': models_data,
            'count': len(models_data)
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@model_families_bp.route('/api/global-totals')
@require_admin_session
def get_global_totals():
    """Get global totals for all models across the system"""
    try:
        global_totals = model_manager.get_global_totals()
        
        return jsonify({
            'success': True,
            'totals': global_totals
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500