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
                    'cat_personality': model.cat_personality
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
            cat_personality=data.get('cat_personality', 'curious')
        )
        
        # Add the model
        auto_whitelist = data.get('auto_whitelist', True)
        success = model_manager.add_custom_model(model_info, auto_whitelist)
        
        if success:
            # Log admin action asynchronously (non-blocking)
            import threading
            def background_log():
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
            
            log_thread = threading.Thread(target=background_log)
            log_thread.daemon = True
            log_thread.start()
            
            return jsonify({
                'success': True,
                'message': f'Model {model_info.display_name} added successfully',
                'model_id': model_info.model_id,
                'whitelisted': auto_whitelist
            })
        else:
            return jsonify({'error': 'Failed to add model (already exists?)'}), 400
            
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
    
    # Remove model_id from validation data to prevent uniqueness check
    validation_data = data.copy()
    validation_data.pop('model_id', None)
    
    # Validate the model data (excluding model_id)
    errors = model_manager.validate_model_data(validation_data)
    if errors:
        return jsonify({'error': 'Validation failed', 'field_errors': errors}), 400
    
    try:
        # Convert numeric fields
        updates = {}
        for field in ['input_cost_per_1m', 'output_cost_per_1m', 'context_length']:
            if field in data:
                updates[field] = float(data[field]) if field != 'context_length' else int(data[field])
        
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
    """Delete a custom model"""
    try:
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