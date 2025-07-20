#!/usr/bin/env python3
"""
Quick cleanup script to reset all user request limits to unlimited (null)
This removes any previously set token-based limits from the database.

Usage: python cleanup_user_limits.py
"""

import sys
import os

# Validate environment variables
required_env_vars = ['FIREBASE_DATABASE_URL', 'GOOGLE_APPLICATION_CREDENTIALS']
for var in required_env_vars:
    if not os.environ.get(var):
        print(f"❌ Error: Environment variable {var} is not set!")
        print(f"💡 Please run this script via the cleanup_user_limits.bat file")
        sys.exit(1)

print(f"🔥 Firebase URL: {os.environ.get('FIREBASE_DATABASE_URL')}")
print(f"🔑 Service Key: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
print()

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import after environment variables are set
try:
    from services.user_store import user_store
    print("✅ Successfully connected to Firebase!")
    print()
except Exception as e:
    print(f"❌ Failed to connect to Firebase: {e}")
    print(f"💡 Check your Firebase credentials and network connection")
    sys.exit(1)

def main():
    """Reset all user limits to unlimited"""
    
    print("🧹 Starting user limits cleanup...")
    print(f"📊 Total users in system: {user_store.get_user_count()}")
    
    updated_count = 0
    
    # Get all users and reset their limits
    with user_store.lock.write_lock():
        for token, user in user_store.users.items():
            # Check if user has any limits set
            has_limits = any(limit is not None for limit in user.prompt_limits.values()) if user.prompt_limits else False
            
            if has_limits:
                print(f"🔄 Resetting limits for user {token[:8]}... (Type: {user.type.value}, Nickname: {user.nickname or 'None'})")
                
                # Show current limits
                current_limits = {}
                for family, limit in user.prompt_limits.items():
                    if limit is not None:
                        current_limits[family] = limit
                
                if current_limits:
                    print(f"   📋 Current limits: {current_limits}")
                
                # Reset all limits to None (unlimited)
                user.prompt_limits = {
                    'openai': None,
                    'anthropic': None, 
                    'google': None,
                    'mistral': None
                }
                
                # Add to flush queue to save changes to Firebase
                with user_store.flush_queue_lock:
                    user_store.flush_queue.add(token)
                
                updated_count += 1
                print(f"   ✅ Limits reset to unlimited")
            else:
                print(f"ℹ️  User {token[:8]}... already has unlimited limits (Type: {user.type.value})")
    
    print(f"\n🎉 Cleanup completed!")
    print(f"📈 Users updated: {updated_count}")
    print(f"📊 Users unchanged: {user_store.get_user_count() - updated_count}")
    print(f"🔥 Changes queued for Firebase sync")
    
    # Force immediate flush to Firebase
    print(f"\n🚀 Flushing changes to Firebase...")
    flush_count = 0
    with user_store.flush_queue_lock:
        flush_tokens = list(user_store.flush_queue.copy())
        user_store.flush_queue.clear()
    
    if flush_tokens:
        print(f"💾 Saving {len(flush_tokens)} users to Firebase...")
        for i in range(0, len(flush_tokens), 10):  # Process in batches of 10
            batch = flush_tokens[i:i + 10]
            user_store._batch_flush_to_firebase(batch)
            flush_count += len(batch)
            print(f"   💾 Batch {i//10 + 1}: {len(batch)} users saved")
    
    print(f"\n✨ All done! {flush_count} users synced to Firebase")
    print(f"🗑️ You can now delete this cleanup script: cleanup_user_limits.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)