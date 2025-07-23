import json
import threading
import time
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import hashlib

# Google Sheets API imports
def try_import_gspread():
    """Try to import gspread and install if missing"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        print("[OK] Google Sheets integration available")
        return True, gspread, Credentials
    except ImportError as e:
        print(f"[FAIL] Google Sheets integration not available: {e}")
        print("[TOOL] Attempting to install missing packages...")
        
        try:
            import subprocess
            import sys
            
            # Try to install the missing packages
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "gspread>=6.2.1", "google-auth>=2.40.3", 
                "--quiet", "--disable-pip-version-check"
            ])
            
            print(" Packages installed, attempting to import again...")
            
            # Try importing again
            import gspread
            from google.oauth2.service_account import Credentials
            print("[OK] Google Sheets integration now available after installation")
            return True, gspread, Credentials
            
        except subprocess.CalledProcessError as install_error:
            print(f"[FAIL] Failed to auto-install packages: {install_error}")
        except ImportError as import_error:
            print(f"[FAIL] Still can't import after installation: {import_error}")
        except Exception as other_error:
            print(f"[FAIL] Unexpected error during auto-installation: {other_error}")
        
        return False, None, None

# Attempt to import and auto-install if needed
GSPREAD_AVAILABLE, gspread, Credentials = try_import_gspread()

# Fallback if auto-installation failed
if not GSPREAD_AVAILABLE:
    print("[ERROR] Google Sheets integration unavailable.")
    print(" Manual installation: pip install gspread google-auth")
    gspread = None
    Credentials = None

# Firebase imports
try:
    import firebase_admin
    from firebase_admin import db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

@dataclass
class ConversationEntry:
    """Single conversation message entry"""
    timestamp: str
    user_token: str
    user_nickname: str
    model_family: str
    model_name: str
    input_text: str
    output_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    ip_hash: str
    user_agent: Optional[str]
    session_id: Optional[str]
    conversation_id: Optional[str]
    
    def to_row(self) -> List[str]:
        """Convert to spreadsheet row format - simplified 4-column format"""
        return [
            self.input_text,   # A column: Input
            self.output_text,  # B column: Output  
            self.timestamp,    # C column: Timestamp
            self.model_name    # D column: Model Name
        ]

class GoogleSheetsLogger:
    """Google Sheets logging backend"""
    
    def __init__(self, service_account_key: str, spreadsheet_id: str):
        self.service_account_key = service_account_key
        self.spreadsheet_id = spreadsheet_id
        self.client = None
        self.spreadsheet = None
        self.worksheets = {}  # Cache worksheets by name
        self.lock = threading.Lock()
        self._initialize_client()
    
    def _sanitize_service_account_json(self, json_str: str) -> str:
        """Sanitize service account JSON to handle common formatting issues"""
        try:
            # First, try to parse as-is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # If parsing fails, try to fix common issues
            
            # Replace literal newlines in private_key with \n
            import re
            
            # Fix private keys that have literal newlines instead of \n
            def fix_private_key(match):
                key_content = match.group(1)
                # Replace any actual newlines with \n
                key_content = key_content.replace('\n', '\\n')
                return f'"private_key": "{key_content}"'
            
            # Apply the fix
            sanitized = re.sub(
                r'"private_key"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                fix_private_key,
                json_str,
                flags=re.DOTALL
            )
            
            # Test if the fix worked
            try:
                json.loads(sanitized)
                print("[OK] Google Sheets: Auto-fixed JSON formatting issues")
                return sanitized
            except json.JSONDecodeError as e:
                print(f"[ERROR] Google Sheets: Could not auto-fix JSON: {e}")
                return json_str
    
    def _initialize_client(self):
        """Initialize Google Sheets client"""
        if not GSPREAD_AVAILABLE:
            print("[ERROR] Google Sheets: gspread not available")
            return False
        
        try:
            # Parse service account credentials with sanitization
            if isinstance(self.service_account_key, str):
                sanitized_json = self._sanitize_service_account_json(self.service_account_key)
                creds_dict = json.loads(sanitized_json)
            else:
                creds_dict = self.service_account_key
            
            print("[CHART] Google Sheets: Service account JSON parsed successfully")
            
            # Set up credentials with required scopes
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            print("[CHART] Google Sheets: Credentials created successfully")
            
            # Initialize gspread client
            self.client = gspread.authorize(credentials)
            print("[CHART] Google Sheets: Client authorized successfully")
            
            # Try to open the spreadsheet
            self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
            print(f"[CHART] Google Sheets: Successfully connected to spreadsheet '{self.spreadsheet.title}'")
            return True
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Google Sheets: Invalid service account JSON - {e}")
            self.client = None
            self.spreadsheet = None
            return False
        except gspread.SpreadsheetNotFound as e:
            print(f"[ERROR] Google Sheets: Spreadsheet not found or no access - {e}")
            print(f"[ERROR] Check: 1) Spreadsheet ID is correct, 2) Service account has access to the sheet")
            self.client = None
            self.spreadsheet = None
            return False
        except Exception as e:
            print(f"[ERROR] Google Sheets: Failed to initialize client - {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            self.client = None
            self.spreadsheet = None
            return False
    
    def get_or_create_worksheet(self, user_identifier: str, nickname: str = None) -> Optional[Any]:
        """Get or create worksheet for user (using nickname but fallback to token)"""
        if not self.client or not self.spreadsheet:
            return None
        
        # Use nickname if available, otherwise use user token for sheet name
        identifier = nickname if nickname and nickname != "Anonymous" else user_identifier
        sheet_name = self._sanitize_sheet_name(identifier)
        
        with self.lock:
            # Check cache first
            if sheet_name in self.worksheets:
                return self.worksheets[sheet_name]
            
            try:
                # Try to get existing worksheet
                worksheet = self.spreadsheet.worksheet(sheet_name)
                print(f"[CHART] Google Sheets: Found existing sheet '{sheet_name}'")
                
            except gspread.WorksheetNotFound:
                # Create new worksheet
                try:
                    worksheet = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=4)
                    print(f"[CHART] Google Sheets: Created new sheet '{sheet_name}'")
                    
                    # Add headers - simplified 4-column format
                    headers = ["Input", "Output", "Timestamp", "Model Name"]
                    worksheet.append_row(headers)
                    
                    # Format header row
                    worksheet.format('A1:D1', {
                        'backgroundColor': {'red': 0.9, 'green': 0.9, 'blue': 0.9},
                        'textFormat': {'bold': True}
                    })
                    
                except Exception as e:
                    print(f"[ERROR] Google Sheets: Failed to create sheet '{sheet_name}' - {e}")
                    return None
            
            # Cache the worksheet
            self.worksheets[sheet_name] = worksheet
            return worksheet
    
    def _sanitize_sheet_name(self, nickname: str) -> str:
        """Sanitize nickname for Google Sheets worksheet name"""
        if not nickname or nickname == "Anonymous":
            nickname = "Anonymous_Users"
        
        # Google Sheets name restrictions: max 100 chars, no special chars
        sanitized = "".join(c for c in nickname if c.isalnum() or c in " -_")[:50]
        if not sanitized:
            sanitized = "Unknown_User"
        
        return sanitized
    
    def log_conversation(self, entry: ConversationEntry) -> bool:
        """Log a conversation entry to appropriate worksheet"""
        if not self.client or not self.spreadsheet:
            return False
        
        try:
            worksheet = self.get_or_create_worksheet(entry.user_token, entry.user_nickname)
            if not worksheet:
                return False
            
            # Add the row
            worksheet.append_row(entry.to_row())
            return True
            
        except Exception as e:
            print(f"[ERROR] Google Sheets: Failed to log conversation - {e}")
            return False
    
    def batch_log_conversations(self, entries: List[ConversationEntry]) -> bool:
        """Batch log multiple conversation entries"""
        if not entries or not self.client or not self.spreadsheet:
            return False
        
        try:
            # Group entries by user identifier for efficient batching
            entries_by_user = {}
            for entry in entries:
                # Use nickname if available, otherwise use user token
                user_key = entry.user_nickname if entry.user_nickname and entry.user_nickname != "Anonymous" else entry.user_token
                if user_key not in entries_by_user:
                    entries_by_user[user_key] = []
                entries_by_user[user_key].append(entry)
            
            # Batch append to each worksheet
            for user_key, user_entries in entries_by_user.items():
                # Get the first entry to extract user info for worksheet creation
                first_entry = user_entries[0]
                worksheet = self.get_or_create_worksheet(first_entry.user_token, first_entry.user_nickname)
                if worksheet:
                    rows = [entry.to_row() for entry in user_entries]
                    worksheet.append_rows(rows)
                    print(f"[CHART] Google Sheets: Batched {len(rows)} entries for '{user_key}'")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Google Sheets: Failed to batch log conversations - {e}")
            return False

class ConversationLogger:
    """Main conversation logging service with queue-based processing"""
    
    def __init__(self):
        self.enabled = False  # Start disabled to prevent 500 errors
        self.log_input = True
        self.log_output = True
        self.last_number_of_chars = 5000
        self.max_output_length = 5000
        self.retention_days = 30
        
        # Google Sheets backend
        self.sheets_logger: Optional[GoogleSheetsLogger] = None
        
        # Queue-based logging for performance
        self.log_queue = Queue(maxsize=1000)
        self.batch_size = 10
        self.batch_timeout = 30  # seconds
        
        # Worker thread
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_logged': 0,
            'failed_logs': 0,
            'queue_full_errors': 0,
            'last_log_time': None,
            'sheets_connected': False
        }
        self.stats_lock = threading.Lock()
        
        # Firebase configuration persistence
        self.firebase_db = None
        self.config_key = 'configs/conversation_logging'
        
        # Initialize Firebase and load saved configuration
        self._initialize_firebase()
        self._load_config_from_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection for configuration persistence"""
        if not FIREBASE_AVAILABLE:
            print("Firebase not available for conversation logging config persistence")
            return
        
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                self.firebase_db = db.reference()
                print("Using existing Firebase connection for conversation logging")
            else:
                print("Firebase not initialized yet - will attempt to connect later")
        except Exception as e:
            print(f"Failed to connect to Firebase for conversation logging config: {e}")
            self.firebase_db = None
    
    def reconnect_firebase(self):
        """Attempt to reconnect to Firebase if it wasn't available during initialization"""
        if not self.firebase_db and FIREBASE_AVAILABLE:
            try:
                if firebase_admin._apps:
                    self.firebase_db = db.reference()
                    print(" Successfully connected to Firebase for conversation logging")
                    self._load_config_from_firebase()
                    return True
                else:
                    print("[ERROR] Firebase apps not initialized yet for conversation logging")
            except Exception as e:
                print(f"[FAIL] Failed to reconnect to Firebase: {e}")
        elif not FIREBASE_AVAILABLE:
            print("[ERROR] Firebase not available (firebase-admin not installed)")
        elif self.firebase_db:
            print(" Firebase already connected for conversation logging")
        return bool(self.firebase_db)
    
    def _load_config_from_firebase(self):
        """Load conversation logging configuration from Firebase"""
        if not self.firebase_db:
            print("[ERROR] Firebase not available - cannot load conversation logging config")
            return
        
        try:
            print(f" Loading conversation logging config from Firebase at: {self.config_key}")
            config_ref = self.firebase_db.child(self.config_key)
            saved_config = config_ref.get()
            
            if saved_config:
                print(" Found saved conversation logging configuration in Firebase")
                print(f" Config keys found: {list(saved_config.keys())}")
                
                # Load basic settings
                self.enabled = saved_config.get('enabled', False)
                self.log_input = saved_config.get('log_input', True)
                self.log_output = saved_config.get('log_output', True)
                self.last_number_of_chars = saved_config.get('last_number_of_chars', saved_config.get('max_input_length', 5000))  # Backward compatibility
                self.max_output_length = saved_config.get('max_output_length', 5000)
                self.retention_days = saved_config.get('retention_days', 30)
                
                # Load Google Sheets configuration if present
                sheets_config = saved_config.get('google_sheets')
                if sheets_config:
                    print(f"[CHART] Found Google Sheets config: {list(sheets_config.keys())}")
                    if GSPREAD_AVAILABLE:
                        spreadsheet_id = sheets_config.get('spreadsheet_id')
                        print(f"[CHART] Spreadsheet ID: {spreadsheet_id}")
                        
                        # Try to load base64 encoded key first (new format)
                        service_account_key_b64 = sheets_config.get('service_account_key_b64')
                        service_account_key = None
                        
                        if service_account_key_b64:
                            print(f"[CHART] Found base64 service account key ({len(service_account_key_b64)} chars)")
                            try:
                                # Decode from base64
                                service_account_key = base64.b64decode(service_account_key_b64).decode('utf-8')
                                print("[CHART] Successfully decoded base64 service account key")
                            except Exception as e:
                                print(f"[FAIL] Failed to decode base64 service account key: {e}")
                        
                        # Fallback to old format (plain text) for backward compatibility
                        if not service_account_key:
                            service_account_key = sheets_config.get('service_account_key')
                            if service_account_key:
                                print("[CHART] Using plain text service account key (fallback)")
                        
                        if spreadsheet_id and service_account_key:
                            try:
                                print("[CHART] Initializing Google Sheets logger...")
                                self.sheets_logger = GoogleSheetsLogger(service_account_key, spreadsheet_id)
                                with self.stats_lock:
                                    self.stats['sheets_connected'] = bool(self.sheets_logger.client)
                                print("[OK] Google Sheets logger initialized successfully")
                            except Exception as e:
                                print(f"[FAIL] Failed to initialize Google Sheets from saved config: {e}")
                        else:
                            print("[FAIL] Missing spreadsheet_id or service_account_key")
                    else:
                        print("[FAIL] Google Sheets not available (gspread not installed)")
                else:
                    print("[CHART] No Google Sheets config found in saved configuration")
                
                # Start worker thread if enabled
                if self.enabled and not self.worker_thread:
                    self.start_worker()
                
                print(f"[OK] Loaded conversation logging config - Enabled: {self.enabled}, Sheets: {bool(self.sheets_logger)}")
            else:
                print(" No saved conversation logging configuration found in Firebase")
                
        except Exception as e:
            print(f"[FAIL] Failed to load conversation logging config from Firebase: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_config_to_firebase(self):
        """Save current conversation logging configuration to Firebase"""
        if not self.firebase_db:
            print("[ERROR] Firebase not available - cannot save conversation logging config")
            return False
        
        try:
            config_data = {
                'enabled': self.enabled,
                'log_input': self.log_input,
                'log_output': self.log_output,
                'last_number_of_chars': self.last_number_of_chars,
                'max_output_length': self.max_output_length,
                'retention_days': self.retention_days,
                'last_updated': datetime.now().isoformat()
            }
            
            # Include Google Sheets config if present (even if logging is disabled)
            if self.sheets_logger:
                print(" Google Sheets logger exists, preparing to save configuration...")
                try:
                    # Base64 encode the service account key for safe Firebase storage
                    service_account_key_b64 = base64.b64encode(
                        self.sheets_logger.service_account_key.encode('utf-8')
                    ).decode('utf-8')
                    
                    config_data['google_sheets'] = {
                        'spreadsheet_id': self.sheets_logger.spreadsheet_id,
                        'service_account_key_b64': service_account_key_b64,
                        'connected': bool(self.sheets_logger.client)
                    }
                    
                    print(f" Saving Google Sheets config - Spreadsheet ID: {self.sheets_logger.spreadsheet_id}")
                    print(f" Service account key length: {len(service_account_key_b64)} chars (base64)")
                    print(f" Google Sheets connected: {bool(self.sheets_logger.client)}")
                except Exception as e:
                    print(f"[FAIL] Error preparing Google Sheets config for save: {e}")
            else:
                print(" No Google Sheets logger configured - skipping Google Sheets config")
            
            print(f" Saving conversation logging config to Firebase at: {self.config_key}")
            print(f" Config data keys: {list(config_data.keys())}")
            
            config_ref = self.firebase_db.child(self.config_key)
            config_ref.set(config_data)
            
            print("[OK] Conversation logging configuration saved to Firebase successfully")
            return True
            
        except Exception as e:
            print(f"[FAIL] Failed to save conversation logging config to Firebase: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def configure(self, config: Dict[str, Any]) -> bool:
        """Configure the conversation logger"""
        try:
            print(f"[TOOL] Configuring conversation logger with config keys: {list(config.keys())}")
            
            # Only update enabled if explicitly provided, otherwise keep current value
            if 'enabled' in config:
                self.enabled = config['enabled']
                print(f"[TOOL] Updated enabled status to: {self.enabled}")
            
            self.log_input = config.get('log_input', self.log_input)
            self.log_output = config.get('log_output', self.log_output)
            self.last_number_of_chars = config.get('last_number_of_chars', config.get('max_input_length', self.last_number_of_chars))  # Backward compatibility
            self.max_output_length = config.get('max_output_length', self.max_output_length)
            self.retention_days = config.get('retention_days', self.retention_days)
            
            print(f"[TOOL] Current configuration - Enabled: {self.enabled}, Input: {self.log_input}, Output: {self.log_output}")
            
            # Configure Google Sheets if provided (regardless of enabled state)
            if config.get('google_sheets'):
                print("[TOOL] Configuring Google Sheets...")
                sheets_config = config['google_sheets']
                service_account_key = sheets_config.get('service_account_key')
                spreadsheet_id = sheets_config.get('spreadsheet_id')
                
                print(f"[TOOL] Spreadsheet ID provided: {bool(spreadsheet_id)}")
                print(f"[TOOL] Service account key provided: {bool(service_account_key)}")
                
                if service_account_key and spreadsheet_id:
                    try:
                        print("[TOOL] Creating GoogleSheetsLogger...")
                        self.sheets_logger = GoogleSheetsLogger(service_account_key, spreadsheet_id)
                        
                        with self.stats_lock:
                            self.stats['sheets_connected'] = bool(self.sheets_logger.client)
                        
                        if self.sheets_logger.client:
                            print(f"[OK] Google Sheets logger created and connected successfully")
                            print(f"[CHART] Spreadsheet: {getattr(self.sheets_logger.spreadsheet, 'title', 'Unknown')}")
                        else:
                            print(f"[FAIL] Google Sheets logger created but connection failed")
                            print(f"[TOOL] Check: 1) Spreadsheet ID is correct, 2) Service account has access, 3) Dependencies installed")
                    except Exception as e:
                        print(f"[FAIL] Failed to create Google Sheets logger: {e}")
                        return False
                else:
                    print("[ERROR] Conversation Logger: Missing Google Sheets spreadsheet_id or service_account_key")
                    return False
            
            # Start worker thread if enabled
            if self.enabled and not self.worker_thread:
                self.start_worker()
            elif not self.enabled and self.worker_thread:
                self.stop_worker()
            
            print(f"Conversation Logger: {'Enabled' if self.enabled else 'Disabled'}")
            if self.enabled:
                print(f"Conversation Logger: Input={self.log_input}, Output={self.log_output}")
                print(f"Conversation Logger: Last chars: {self.last_number_of_chars}, Max output length: {self.max_output_length}")
                print(f"Conversation Logger: Retention={self.retention_days} days")
            
            # Save configuration to Firebase for persistence
            self._save_config_to_firebase()
            
            return True
            
        except Exception as e:
            print(f"Conversation Logger: Configuration failed - {e}")
            return False
    
    def start_worker(self):
        """Start the background worker thread"""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        
        self.shutdown_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("Conversation Logger: Worker thread started")
    
    def stop_worker(self):
        """Stop the background worker thread"""
        if self.worker_thread:
            self.shutdown_event.set()
            self.worker_thread.join(timeout=10)
            print("Conversation Logger: Worker thread stopped")
    
    def _worker_loop(self):
        """Background worker thread loop"""
        batch = []
        last_batch_time = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Try to get an item from queue with timeout
                try:
                    entry = self.log_queue.get(timeout=5)
                    batch.append(entry)
                    self.log_queue.task_done()
                except Empty:
                    pass
                
                current_time = time.time()
                
                # Process batch if it's full or timeout reached
                should_process = (
                    len(batch) >= self.batch_size or 
                    (batch and (current_time - last_batch_time) >= self.batch_timeout)
                )
                
                if should_process and batch:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = current_time
                
            except Exception as e:
                print(f"Conversation Logger: Worker error - {e}")
                time.sleep(5)
        
        # Process remaining batch on shutdown
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[ConversationEntry]):
        """Process a batch of conversation entries"""
        if not batch:
            return
        
        try:
            success = False
            
            # Try Google Sheets logging
            if self.sheets_logger:
                success = self.sheets_logger.batch_log_conversations(batch)
            
            with self.stats_lock:
                if success:
                    self.stats['total_logged'] += len(batch)
                    self.stats['last_log_time'] = datetime.now().isoformat()
                else:
                    self.stats['failed_logs'] += len(batch)
            
            print(f"Conversation Logger: Processed batch of {len(batch)} entries - {'Success' if success else 'Failed'}")
            
        except Exception as e:
            print(f"Conversation Logger: Batch processing failed - {e}")
            with self.stats_lock:
                self.stats['failed_logs'] += len(batch)
    
    def log_conversation(self, user_token: str, user_nickname: str, model_family: str,
                        model_name: str, input_text: str, output_text: str,
                        input_tokens: int, output_tokens: int, cost_usd: float,
                        ip_address: str, user_agent: str = None,
                        session_id: str = None, conversation_id: str = None) -> bool:
        """Log a conversation entry"""
        if not self.enabled:
            return False
        
        try:
            # Apply privacy filters
            logged_input = input_text if self.log_input else "[Input logging disabled]"
            logged_output = output_text if self.log_output else "[Output logging disabled]"
            
            # Apply length limits
            # Apply last N characters limit for input
            if self.last_number_of_chars and len(logged_input) > self.last_number_of_chars:
                logged_input = "...[truncated]" + logged_input[-self.last_number_of_chars:]
            if len(logged_output) > self.max_output_length:
                logged_output = logged_output[:self.max_output_length] + "...[truncated]"
            
            # Hash IP for privacy
            ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()
            
            # Create conversation entry
            entry = ConversationEntry(
                timestamp=datetime.now().isoformat(),
                user_token=user_token,
                user_nickname=user_nickname or "Anonymous",
                model_family=model_family,
                model_name=model_name,
                input_text=logged_input,
                output_text=logged_output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost_usd,
                ip_hash=ip_hash,
                user_agent=user_agent,
                session_id=session_id,
                conversation_id=conversation_id
            )
            
            # Add to queue for background processing
            try:
                self.log_queue.put_nowait(entry)
                return True
            except:
                # Queue is full
                with self.stats_lock:
                    self.stats['queue_full_errors'] += 1
                print("Warning: Conversation Logger: Queue full, dropping log entry")
                return False
                
        except Exception as e:
            print(f"Conversation Logger: Failed to log conversation - {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        with self.stats_lock:
            return {
                **self.stats,
                'enabled': self.enabled,
                'queue_size': self.log_queue.qsize(),
                'log_input': self.log_input,
                'log_output': self.log_output,
                'last_number_of_chars': self.last_number_of_chars,
                'max_output_length': self.max_output_length,
                'retention_days': self.retention_days
            }
    
    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """Update logger settings dynamically"""
        try:
            old_enabled = self.enabled
            
            # Update settings
            self.enabled = settings.get('enabled', self.enabled)
            self.log_input = settings.get('log_input', self.log_input)
            self.log_output = settings.get('log_output', self.log_output)
            self.last_number_of_chars = settings.get('last_number_of_chars', settings.get('max_input_length', self.last_number_of_chars))  # Backward compatibility
            self.max_output_length = settings.get('max_output_length', self.max_output_length)
            
            # Handle worker thread state changes
            if old_enabled != self.enabled:
                if self.enabled and not self.worker_thread:
                    self.start_worker()
                elif not self.enabled and self.worker_thread:
                    self.stop_worker()
            
            print(f"Conversation Logger: Settings updated - Enabled: {self.enabled}")
            
            # Save updated settings to Firebase for persistence
            self._save_config_to_firebase()
            
            return True
            
        except Exception as e:
            print(f"Conversation Logger: Failed to update settings - {e}")
            return False

# Global conversation logger instance
conversation_logger = ConversationLogger()