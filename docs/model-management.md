# 🧬 Model Management Guide

NyanProxy provides a comprehensive model management system that allows you to dynamically add, configure, and control AI models without modifying code or environment variables.

## 🎯 Overview

The Model Management system provides:
- **Dynamic Model Addition**: Add new models via admin interface
- **Provider Support**: OpenAI, Anthropic, Google, Mistral, Groq, Cohere
- **Security**: Whitelist-only access control
- **Cost Tracking**: Per-model usage and cost analytics
- **Firebase Persistence**: Automatic synchronization across deployments

## 🚀 Quick Start

### Accessing Model Management

1. Navigate to your NyanProxy admin panel: `/admin/login`
2. Enter your admin credentials
3. Click "🧬 Model Families" in the navigation menu

### Adding a New Model

1. Click **"➕ Add New Model"** button
2. Fill in the model details:
   - **Model ID**: `gpt-4.1-turbo` (API identifier)
   - **Provider**: Select from dropdown (OpenAI, Anthropic, etc.)
   - **Display Name**: `GPT-4.1 Turbo` (human-readable name)
   - **Description**: Brief description of capabilities
   - **Costs**: Input/output cost per 1K tokens
   - **Context Length**: Maximum tokens (e.g., 128000)
   - **Cat Personality**: Fun theme (curious, playful, smart, etc.)
   - **Capabilities**: Streaming, function calling, premium status
3. Check **"Auto-enable when added"** to immediately whitelist the model
4. Click **"Add Model"**

The model is now available for API requests!

## 🔧 Managing Models

### Default Models

NyanProxy comes with these pre-configured models:

**OpenAI Models:**
- `gpt-4o` - Most advanced GPT-4 model
- `gpt-4o-mini` - Smaller, faster GPT-4 model  
- `gpt-3.5-turbo` - Fast and efficient for most tasks

**Anthropic Models:**
- `claude-3-5-sonnet-20241022` - Most intelligent Claude model
- `claude-3-haiku-20240307` - Fastest Claude model

**Google Models:**
- `gemini-1.5-pro` - Google's most capable model
- `gemini-1.5-flash` - Fast and efficient Gemini model

### Model Controls

**Individual Model Toggle:**
- Use the toggle switches to enable/disable specific models
- Changes are applied immediately

**Bulk Operations:**
- **Enable All Models**: Enable all models across all providers
- **Disable All Models**: Disable all models (use with caution!)
- **Provider Controls**: Enable/disable all models for a specific provider

### Custom Models Management

1. Click **"🔧 Manage Custom Models"** to view all custom models
2. See detailed information: costs, context length, status
3. **Delete**: Remove custom models you no longer need

## 🔒 Security & Whitelisting

### How Whitelisting Works

NyanProxy uses a **whitelist-only** security model:

1. **Model Must Exist**: The model must be defined in the system
2. **Model Must Be Whitelisted**: The model must be explicitly enabled
3. **Request Validation**: Every API request checks both conditions

### Security Examples

```
✅ ALLOWED: "gpt-4o" (exists + whitelisted)
❌ BLOCKED: "o3-mini" (doesn't exist)
❌ BLOCKED: "custom-model" (exists but not whitelisted)
```

### API Response for Blocked Models

```json
{
  "error": {
    "message": "Model 'o3-mini' is not whitelisted for use",
    "type": "model_not_allowed"
  }
}
```

## 📊 Usage Analytics

### Cost Tracking

Each model tracks:
- **Total Requests**: Number of API calls
- **Token Usage**: Input/output tokens consumed
- **Cost**: Calculated cost based on token usage
- **Success Rate**: Percentage of successful requests

### Viewing Analytics

1. Navigate to **"📊 Usage Analytics"** from the Model Families page
2. View:
   - Top models by cost
   - Top models by requests
   - Per-model performance metrics
   - Cost breakdown by provider

## 🔧 Advanced Configuration

### Model Properties

When adding a model, you can configure:

**Required Fields:**
- `model_id`: API identifier (e.g., "gpt-4.1-turbo")
- `provider`: AI service provider
- `display_name`: Human-readable name
- `description`: Model capabilities description

**Optional Fields:**
- `input_cost_per_1k`: Cost per 1K input tokens ($)
- `output_cost_per_1k`: Cost per 1K output tokens ($)
- `context_length`: Maximum context window (default: 4096)
- `supports_streaming`: Real-time response support
- `supports_function_calling`: Function/tool calling capability
- `is_premium`: Premium model designation
- `cat_personality`: Theme for fun categorization

### Cat Personalities

Choose from these fun themes:
- 🙀 **Curious**: Inquisitive models
- 😸 **Playful**: Fun, creative models
- 😴 **Sleepy**: Relaxed, casual models
- 😾 **Grumpy**: Serious, no-nonsense models
- 🤓 **Smart**: Intelligent, analytical models
- 💨 **Fast**: Quick, efficient models

## 🗄️ Data Persistence

### Firebase Integration

When Firebase is configured:
- Custom models sync automatically across deployments
- Model configurations persist through restarts
- Real-time updates across multiple instances

### Local File Fallback

Without Firebase:
- Models saved to `custom_models.json`
- Configuration persists locally
- Manual backup recommended

## 🛠️ Troubleshooting

### Common Issues

**Model Not Appearing:**
1. Check if model was added successfully
2. Verify model is whitelisted (toggle on)
3. Refresh the page

**API Requests Failing:**
1. Confirm model exists in system
2. Verify model is whitelisted
3. Check API key is configured for the provider

**Permission Errors:**
1. Ensure you're logged in as admin
2. Check admin key is correct
3. Verify session hasn't expired

### Getting Help

If you encounter issues:
1. Check the browser console for JavaScript errors
2. Review the NyanProxy logs for backend errors
3. Verify your Firebase configuration (if using)
4. Test with default models first

## 🎉 Best Practices

### Model Naming

- Use clear, descriptive model IDs
- Follow provider conventions (e.g., `gpt-4.1-turbo`)
- Avoid special characters or spaces

### Cost Management

- Set realistic cost estimates for budget tracking
- Monitor usage analytics regularly
- Consider token limits for expensive models

### Security

- Only whitelist models you trust
- Regularly review enabled models
- Use the "Disable All" feature for maintenance

### Performance

- Enable streaming for better user experience
- Set appropriate context lengths
- Monitor response times via analytics

---

*Need help? Check out the [main documentation](../README.md) or create an issue on the repository.*