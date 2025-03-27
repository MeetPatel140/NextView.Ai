#!/usr/bin/env python

import os
import logging
import redis
import time
import importlib
from celery import Celery
from celery.signals import worker_ready, worker_init

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
)
logger = logging.getLogger(__name__)

# Check Redis connection with retry mechanism
def check_redis_connection(max_retries=3, retry_delay=2):
    """
    Check if Redis is available with retry mechanism
    Returns True if connection successful, False otherwise
    """
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_db = int(os.environ.get('REDIS_DB', 0))
    
    logger.info(f"Checking Redis connection at {redis_host}:{redis_port}/{redis_db}")
    
    for attempt in range(max_retries):
        try:
            # Create Redis client and ping
            r = redis.Redis(host=redis_host, port=redis_port, db=redis_db, socket_timeout=5)
            if r.ping():
                logger.info("Redis connection successful")
                return True
            else:
                logger.warning(f"Redis ping failed on attempt {attempt + 1}/{max_retries}")
        except redis.exceptions.ConnectionError as e:
            logger.warning(f"Redis connection error on attempt {attempt + 1}/{max_retries}: {str(e)}")
        except Exception as e:
            logger.warning(f"Unexpected error checking Redis on attempt {attempt + 1}/{max_retries}: {str(e)}")
        
        if attempt < max_retries - 1:
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error(f"Redis connection failed after {max_retries} attempts")
    return False

# Signal handlers for worker lifecycle events
@worker_init.connect
def init_worker(**kwargs):
    logger.info("Initializing worker...")
    if not check_redis_connection():
        logger.error("Redis connection check failed during worker initialization")
        raise RuntimeError("Redis connection failed during worker initialization")

@worker_ready.connect
def worker_ready_handler(**kwargs):
    logger.info("Worker is ready to receive tasks")

# Add worker configuration
class CeleryWorkerConfig:
    # Broker settings
    broker_connection_retry = True
    broker_connection_max_retries = None  # Retry indefinitely
    broker_connection_timeout = 30
    broker_pool_limit = 10
    
    # Concurrency settings
    worker_concurrency = 4  # Number of worker processes
    worker_prefetch_multiplier = 1  # Prefetch one task at a time
    task_acks_late = True  # Acknowledge tasks after execution
    
    # Task timeouts
    task_time_limit = 300  # 5 minutes max task execution time
    task_soft_time_limit = 240  # 4 minutes soft timeout
    
    # Performance settings
    worker_max_tasks_per_child = 1000  # Restart worker after 1000 tasks
    worker_disable_rate_limits = True  # Disable rate limits for better performance

# Initialize Celery app
celery = Celery('nextview')

# Configure broker URL from environment or use default
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
result_backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Set basic Celery configuration
celery.conf.update(
    broker_url=broker_url,
    result_backend=result_backend,
    broker_connection_retry_on_startup=True,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json'
)

# Apply the config to celery directly
try:
    logger.info("Applying Celery worker configuration")
    celery.config_from_object(CeleryWorkerConfig)
    logger.info("Celery worker configuration applied successfully")
    
    # Import tasks to ensure they are registered with Celery
    try:
        # Import the app package to register tasks
        logger.info("Attempting to import task modules")
        
        # First check if the module exists before importing
        try:
            import app.data_processing
            from app.data_processing import tasks as data_processing_tasks
            importlib.reload(data_processing_tasks)  # Ensure tasks are freshly loaded
            logger.info("Successfully imported data_processing tasks module")
        except ImportError as e:
            logger.warning(f"Could not import data_processing tasks: {str(e)}")
        
        # Verify task registration - but don't fail if no tasks found
        registered_tasks = list(celery.tasks.keys())
        # Filter out internal Celery tasks
        app_tasks = [task for task in registered_tasks if task.startswith('app.')]
        
        if not app_tasks:
            logger.warning("No application tasks were registered automatically")
            # Continue anyway - don't raise an exception
        else:
            logger.info(f"Successfully registered {len(app_tasks)} application tasks: {', '.join(app_tasks)}")
    except Exception as import_error:
        # Log the error but don't raise - allow worker to start anyway
        logger.error(f"Failed to import tasks: {str(import_error)}")
        logger.warning("Continuing worker startup despite task import failure")
    
except Exception as e:
    logger.error(f"Failed to apply Celery configuration: {str(e)}")
    raise

if __name__ == '__main__':
    try:
        logger.info("Starting Celery worker")
        # Set additional worker startup options
        worker_options = [
            'worker',
            '--loglevel=INFO',
            '--pool=prefork',
            f'--concurrency={CeleryWorkerConfig.worker_concurrency}',
            '--without-heartbeat',  # Disable heartbeat for better stability
            '--without-mingle'      # Disable worker sync for faster startup
        ]
        celery.worker_main(worker_options)
    except Exception as e:
        logger.error(f"Failed to start Celery worker: {str(e)}")
        raise