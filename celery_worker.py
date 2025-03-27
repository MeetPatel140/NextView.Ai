from app import create_app, celery

app = create_app()
app_context = app.app_context()
app_context.push()

# Initialize Celery with the Flask app context
celery_app = celery

# This ensures that the Celery worker has access to the Flask application context
# which is necessary for database operations and other Flask extensions
if __name__ == '__main__':
    celery_app.start()