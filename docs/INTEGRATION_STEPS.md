# Integration Steps for AI Key Management System

## Quick Fix for Immediate Retry Logic

To get the retry logic working immediately, you need to make these simple changes to your `core/app.py` file:

### Step 1: Add Import
At the top of `core/app.py`, add this import after the existing imports (around line 50):

```python
from .integrated_api_manager import get_integrated_manager, make_anthropic_call
```

### Step 2: Replace Key Manager Initialization
Find this line (around line 774):
```python
key_manager = APIKeyManager()
```

Replace it with:
```python
# Initialize integrated key manager with retry logic
key_manager = get_integrated_manager()
```

### Step 3: Replace Anthropic Function (Quick Fix)
Find the `anthropic_messages()` function (around line 1945) and replace the retry loop section.

**Find this code block (lines ~1995-2030):**
```python
for attempt in range(max_retries):
    api_key = key_manager.get_api_key('anthropic')
    if not api_key:
        metrics.track_request('chat_completions', time.time() - start_time, error=True)
        return jsonify({"error": "No Anthropic API key configured"}), 500
    
    headers = {
        'x-api-key': api_key,
        'Content-Type': 'application/json',
        'anthropic-version': '2023-06-01'
    }
    
    try:
        session = connection_pool.get_session('anthropic')
        response = session.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=request.json,
            stream=request.json.get('stream', False),
            timeout=getattr(session, '_default_timeout', (10, 30))
        )
        
        response_time = time.time() - start_time
        
        # Handle different response codes
        if response.status_code >= 400:
            error_text = response.text
            if key_manager.handle_api_error('anthropic', api_key, error_text, response.status_code):
                if attempt < max_retries - 1:
                    time.sleep(1)  # Brief pause before retry
                    continue
            else:
                # Error is not retryable, return it immediately
                metrics.track_request('chat_completions', time.time() - start_time, error=True)
                try:
                    error_data = json.loads(error_text)
                    return jsonify(error_data), response.status_code
                except:
                    return jsonify({"error": {"message": error_text, "type": "anthropic_api_error"}}), response.status_code
        else:
            # Success - update key health
            key_manager.update_key_health(api_key, True)
```

**Replace it with this:**
```python
# 🚀 NEW: Use comprehensive retry system
print(f"🔄 Starting Anthropic API call with comprehensive retry system...")

# Execute API call with the new retry system
retry_result = make_anthropic_call(request.json)

if retry_result.success:
    print(f"✅ Anthropic API call successful after {retry_result.attempts_made} attempts")
    print(f"🔑 Keys tried: {', '.join(retry_result.keys_tried)}")
    
    # Extract the response
    api_response = retry_result.response
    response = type('MockResponse', (), {
        'status_code': api_response['status_code'],
        'text': api_response['text'],
        'content': api_response['content'],
        'headers': api_response['headers']
    })()
    
    response_time = time.time() - start_time
else:
    # All retries failed
    print(f"💀 Anthropic API call failed after {retry_result.attempts_made} attempts")
    print(f"💀 Final error: {retry_result.error_message}")
    print(f"🔑 Keys tried: {', '.join(retry_result.keys_tried)}")
    
    metrics.track_request('chat_completions', time.time() - start_time, error=True)
    
    # Return appropriate error response
    status_code = retry_result.final_error_code or 500
    try:
        error_data = json.loads(retry_result.error_message)
        return jsonify(error_data), status_code
    except (json.JSONDecodeError, TypeError):
        return jsonify({
            "error": {
                "message": retry_result.error_message or "All API keys failed",
                "type": "anthropic_api_error",
                "attempts_made": retry_result.attempts_made
            }
        }), status_code
```

### Step 4: Environment Variables
Make sure these environment variables are set:
```bash
# Set max retries (optional, defaults to 3)
export MAX_RETRIES=3

# Your API keys (comma-separated)
export ANTHROPIC_API_KEYS="sk-ant-key1,sk-ant-key2,sk-ant-key3"
export OPENAI_API_KEYS="sk-key1,sk-key2,sk-key3"
```

## Testing the Fix

After making these changes:

1. **Restart your application**
2. **Check the logs** - you should see:
   ```
   🔧 Initializing Integrated API Manager...
   🔧 Loading API keys from environment...
   ✅ Anthropic: X/Y keys ready
   ```

3. **Make an API call** - you should see:
   ```
   🔄 Starting Anthropic API call with comprehensive retry system...
   ✅ Anthropic API call successful after 1 attempts
   🔑 Keys tried: sk-ant-1234...
   ```

4. **Test with invalid keys** - Add a bad key to test removal:
   ```
   export ANTHROPIC_API_KEYS="sk-ant-invalid,sk-ant-good-key"
   ```
   
   You should see:
   ```
   🔴 API Error for anthropic: 401 - Invalid API key
   🔍 Error classified as: invalid_key, should_retry: False
   🗑️ REMOVED key sk-ant-invalid... from anthropic pool (reason: invalid_key)
   🔄 Attempt 2/3 using key sk-ant-good...
   ✅ Anthropic API call successful after 2 attempts
   ```

## What This Gives You

✅ **Real Key Pool Management**: Invalid and quota-exceeded keys are immediately removed  
✅ **Smart Retry Logic**: Tries different keys automatically  
✅ **Proper Error Classification**: Follows your exact specifications  
✅ **Real-time Updates**: Keys removed instantly when they fail  
✅ **Dashboard Analytics**: Comprehensive statistics for monitoring  
✅ **Background Health Checks**: Continuous key validation  

## Monitor the System

After integration, you can monitor the system:

```python
# In your Flask routes, add a health endpoint:
@app.route('/api/key-health')
def key_health():
    from .integrated_api_manager import get_integrated_manager
    manager = get_integrated_manager()
    status = manager.get_comprehensive_status()
    return jsonify(status)
```

Visit `/api/key-health` to see:
- Which keys are healthy vs removed
- Success rates per key
- Error statistics
- Real-time pool status

## Complete Integration (Optional)

For a complete integration that handles all providers (OpenAI, Google, etc.), follow the same pattern:

1. Replace the OpenAI retry loop with `make_openai_call()`
2. Replace the Google retry loop with `make_google_call()`
3. Use the same error handling pattern

This gives you a unified retry system across all AI providers with consistent behavior and monitoring.

## Rollback Instructions

If you need to rollback:

1. Revert the import changes
2. Change back to: `key_manager = APIKeyManager()`
3. Restore the original retry loop code

The system is designed to be a drop-in replacement, so rollback is simple.