# 🗣️ Conversation Logging System

A comprehensive conversation logging system that saves all user interactions to Google Sheets, organized by user nickname.

## 📋 Features

- **Per-User Organization**: Each user gets their own sheet based on their nickname
- **Privacy Controls**: Toggle input/output logging independently  
- **Length Limits**: Configurable max length for inputs and outputs
- **Real-time Dashboard**: Monitor logging status and statistics
- **Background Processing**: Queue-based logging doesn't slow down responses
- **Google Sheets Integration**: Automatic spreadsheet creation and management

## 🚀 Quick Setup

### 1. Create Google Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project or select existing one
3. Enable **Google Sheets API** and **Google Drive API**
4. Go to **Service Accounts** → **Create Service Account**
5. Download the JSON key file
6. Copy the entire JSON content

### 2. Create Google Spreadsheet

1. Create a new [Google Sheet](https://sheets.google.com)
2. Copy the spreadsheet ID from the URL:
   ```
   https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit
   ```
3. Share the spreadsheet with the service account email (found in JSON)

### 3. Configure in Admin Dashboard

1. Go to **Admin Dashboard** → **🗣️ Conversation Logging**
2. Click **⚙️ Configure Sheets**  
3. Enter your **Spreadsheet ID**
4. Paste the **Service Account JSON**
5. Click **Save Configuration**

## 🎛️ Admin Controls

### Toggle Logging
- **Enable/Disable**: Master on/off switch
- **Log Inputs**: Toggle user message logging
- **Log Outputs**: Toggle AI response logging  
- **Max Lengths**: Set character limits (100-10,000)

### Monitor Status
- **Logging Status**: Enabled/Disabled
- **Google Sheets**: Connected/Disconnected
- **Queue Size**: Pending logs to process
- **Total Logged**: Total conversations saved

### Quick Actions
- **🔄 Refresh Stats**: Update statistics
- **📊 Open Google Sheets**: Direct link to spreadsheet
- **📋 View Logs**: Technical logging details

## 📊 Google Sheets Structure

Each user gets their own worksheet with columns:

| Column | Description |
|--------|-------------|
| Timestamp | When the conversation occurred |
| User Token | First 8 characters + "..." |
| Nickname | User's friendly name |
| Model Family | openai, anthropic, google, etc. |
| Model Name | gpt-4, claude-3, etc. |
| Input Text | User's message (truncated if long) |
| Output Text | AI's response (truncated if long) |
| Input Tokens | Token count for input |
| Output Tokens | Token count for output |
| Total Tokens | Sum of input + output |
| Cost (USD) | Estimated cost in dollars |
| IP Hash | Hashed IP for privacy |
| User Agent | Browser/client information |
| Session ID | Session identifier (if available) |
| Conversation ID | Conversation identifier (if available) |

## 🔒 Privacy & Security

### Privacy Features
- **IP Hashing**: IP addresses are SHA-256 hashed
- **Token Truncation**: Only first 8 characters shown
- **Content Limits**: Configurable input/output length limits
- **Optional Logging**: Can disable input or output logging separately

### Security Considerations
- Service account JSON contains sensitive credentials
- Store securely and rotate regularly
- Only grant necessary Google Sheets/Drive permissions
- Consider using environment variables for production

## 🛠️ Technical Details

### Background Processing
- Uses queue-based system for performance
- Batch processing (10 entries at once)
- Background worker thread handles all I/O
- Won't slow down API responses

### Error Handling
- Graceful fallbacks for Google Sheets failures
- Automatic retry for transient errors
- Detailed error logging for debugging
- Queue overflow protection

### Performance
- Asynchronous logging via managed thread pool
- Connection pooling for Google API requests
- Automatic worksheet caching
- Batch operations for efficiency

## 📈 Monitoring

### Dashboard Metrics
- **Queue Size**: Monitor for bottlenecks
- **Failed Logs**: Track error rates
- **Connection Status**: Google Sheets availability
- **Processing Rate**: Logs per minute

### Troubleshooting
1. **"Disconnected" Status**: Check service account permissions
2. **High Queue Size**: Verify Google Sheets connectivity
3. **Missing Logs**: Check conversation logging toggle
4. **Empty Sheets**: Verify user has conversations

## 🔧 Configuration Options

### Environment Variables (Optional)
```bash
CONVERSATION_LOGGING_ENABLED=true
CONVERSATION_LOG_INPUT=true
CONVERSATION_LOG_OUTPUT=true
CONVERSATION_MAX_INPUT_LENGTH=5000
CONVERSATION_MAX_OUTPUT_LENGTH=5000
```

### Programmatic Configuration
```python
from src.services.conversation_logger import conversation_logger

conversation_logger.configure({
    'enabled': True,
    'log_input': True,
    'log_output': True,
    'max_input_length': 5000,
    'max_output_length': 5000,
    'google_sheets': {
        'spreadsheet_id': 'your-spreadsheet-id',
        'service_account_key': 'your-json-key'
    }
})
```

## 🎯 Use Cases

- **User Support**: Review conversation history for support tickets
- **Quality Assurance**: Monitor AI response quality  
- **Analytics**: Analyze usage patterns by user
- **Compliance**: Maintain conversation records
- **Training Data**: Export conversations for model training
- **Debugging**: Trace issues through conversation logs

## ⚠️ Important Notes

- **Storage Costs**: Google Sheets has row limits (10M cells)
- **API Limits**: Google Sheets API has quota limits
- **Large Conversations**: Very long inputs/outputs are truncated
- **Streaming**: Streaming responses show "[Streaming response...]"
- **Retroactive**: Only logs new conversations after enabling

---

🐱 **Happy Logging!** Your conversations are now safely stored and organized by user nickname.