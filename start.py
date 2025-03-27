#!/usr/bin/env python
"""
NextView.AI Startup Script
--------------------------
This script launches all necessary components:
1. Checks Redis connection
2. Starts Celery worker in a separate process
3. Starts Flask application
"""

import os
import sys
import subprocess
import time
import signal
import atexit
import socket
import platform

# Global variables for processes
processes = []

def is_redis_running(host='localhost', port=6379):
    """Check if Redis is running"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False

def start_redis():
    """Start Redis server if not running"""
    if is_redis_running():
        print("✓ Redis is already running")
        return True
    
    # Different commands based on platform
    if platform.system() == 'Windows':
        if os.path.exists('Redis-x64-3.0.504.msi'):
            print("ℹ Redis installer found but Redis is not running.")
            print("Please install and start Redis manually, then run this script again.")
            return False
        else:
            print("⚠ Redis is not installed. Please install Redis and start it manually.")
            return False
    else:
        try:
            print("Starting Redis server...")
            process = subprocess.Popen(['redis-server'], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
            processes.append(process)
            # Give it a moment to start
            time.sleep(2)
            return is_redis_running()
        except:
            print("⚠ Could not start Redis server. Please start it manually.")
            return False

def start_celery():
    """Start Celery worker"""
    print("Starting Celery worker...")
    # Use shell=True on Windows to avoid console window, but not on Unix
    use_shell = platform.system() == 'Windows'
    
    # On Windows, use python -m celery instead of direct celery command
    # This ensures the correct Python interpreter is used
    if platform.system() == 'Windows':
        python_exe = sys.executable
        celery_command = [
            python_exe,
            '-m', 'celery',
            '--app', 'celery_worker.celery',
            'worker',
            '--loglevel=info',
            '--concurrency=1',
            '--pool=solo'
        ]
    else:
        celery_command = [
            'celery',
            '--app', 'celery_worker.celery',
            'worker',
            '--loglevel=info',
            '--concurrency=2',
            '--pool=solo'
        ]
    
    print(f"Running command: {' '.join(celery_command)}")
    
    # Start the worker
    process = subprocess.Popen(
        celery_command,
        stdout=subprocess.PIPE if use_shell else None,
        stderr=subprocess.PIPE if use_shell else None,
        shell=use_shell
    )
    processes.append(process)
    
    # Give Celery time to start
    time.sleep(3)
    
    # Check if it's running
    if process.poll() is not None:
        # Process exited - it failed to start
        print("⚠ Celery worker failed to start. Check celery_worker.py for errors.")
        print("Attempting to start with alternative configuration...")
        
        # Try with different configuration
        if platform.system() == 'Windows':
            celery_command = [
                python_exe,
                '-m', 'celery',
                'worker',
                '--app=celery_worker.celery',
                '--loglevel=debug',  # More verbose logging
                '--pool=solo'
            ]
        else:
            celery_command = [
                'celery',
                'worker',
                '--app=celery_worker.celery',
                '--loglevel=debug',
                '--pool=prefork'
            ]
        
        print(f"Trying alternative command: {' '.join(celery_command)}")
        
        process = subprocess.Popen(
            celery_command,
            stdout=subprocess.PIPE if use_shell else None,
            stderr=subprocess.PIPE if use_shell else None,
            shell=use_shell
        )
        processes.append(process)
        time.sleep(3)
    
    # Final check
    return process.poll() is None  # Still running?

def start_flask():
    """Start Flask application"""
    print("Starting Flask application...")
    
    # Import and run the Flask app in the current process
    from app import create_app
    app = create_app()
    
    # Push an application context
    app_context = app.app_context()
    app_context.push()
    
    # Print access URL
    host = '127.0.0.1'
    port = 5000
    print(f"\n✓ NextView.AI is running!")
    print(f"✓ Access the application at: http://{host}:{port}")
    print("✓ Press Ctrl+C to stop all services\n")
    
    # Start the Flask app
    app.run(debug=True, host=host, port=port, use_reloader=False)

def cleanup():
    """Terminate all processes when the script exits"""
    print("\nShutting down all services...")
    for process in processes:
        try:
            if platform.system() == 'Windows':
                process.terminate()
            else:
                process.send_signal(signal.SIGTERM)
        except:
            pass
    print("All services stopped. Goodbye!")

if __name__ == "__main__":
    # Register cleanup handler
    atexit.register(cleanup)
    
    try:
        # Start Redis if not running
        if not start_redis():
            sys.exit(1)
        
        # Start Celery worker
        if not start_celery():
            print("⚠ Failed to start Celery worker. Please check celery_worker.py.")
            sys.exit(1)
        
        # Start Flask app (this blocks until the app is stopped)
        start_flask()
    
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Stopping services...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting services: {str(e)}")
        sys.exit(1)