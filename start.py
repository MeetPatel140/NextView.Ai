import multiprocessing
import os
import signal
import sys
from time import sleep

def run_flask():
    from run import app
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

def run_celery():
    os.system('celery -A celery_worker.celery worker --loglevel=info --pool=solo')

def signal_handler(signum, frame):
    print('\nShutting down all processes...')
    sys.exit(0)

def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create processes
    flask_process = multiprocessing.Process(target=run_flask)
    celery_process = multiprocessing.Process(target=run_celery)

    try:
        print('Starting Flask application...')
        flask_process.start()
        sleep(2)  # Give Flask a moment to start

        print('Starting Celery worker...')
        celery_process.start()

        # Keep the main process running
        flask_process.join()
        celery_process.join()

    except KeyboardInterrupt:
        print('\nShutting down...')
    finally:
        # Ensure processes are terminated
        if flask_process.is_alive():
            flask_process.terminate()
        if celery_process.is_alive():
            celery_process.terminate()

        flask_process.join()
        celery_process.join()

if __name__ == '__main__':
    main()