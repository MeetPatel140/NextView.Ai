import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from celery import Celery
from config import get_config

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()
celery = Celery()

def create_celery_app(app=None):
    app = app or create_app()
    
    # Configure Celery directly with broker and backend URLs
    celery.conf.broker_url = app.config.get('broker_url')
    celery.conf.result_backend = app.config.get('result_backend')
    
    # Set broker_connection_retry_on_startup to True to fix deprecation warning
    celery.conf.broker_connection_retry_on_startup = True
    
    # Set task_create_missing_queues to ensure tasks are properly routed
    celery.conf.task_create_missing_queues = True
    
    # Configure task serialization
    celery.conf.task_serializer = 'json'
    celery.conf.result_serializer = 'json'
    celery.conf.accept_content = ['json']
    
    # Set task_always_eager to False to ensure tasks are processed by workers
    celery.conf.task_always_eager = False
    
    # Set task_acks_late to True to ensure tasks are acknowledged after execution
    celery.conf.task_acks_late = True
    
    # Define a custom task class that maintains Flask app context
    class ContextTask(celery.Task):
        abstract = True
        
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)
    
    celery.Task = ContextTask
    return celery

def create_app():
    app = Flask(__name__)
    app.config.from_object(get_config())
    
    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    CORS(app)
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Configure login
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    
    # Register blueprints
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from app.api import bp as api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    from app.reports import bp as reports_bp
    app.register_blueprint(reports_bp, url_prefix='/reports')
    
    from app.chatbot import bp as chatbot_bp
    app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
    
    # Initialize Celery
    celery_instance = create_celery_app(app)
    
    # Add context processor for template variables
    @app.context_processor
    def inject_now():
        from datetime import datetime
        return {'now': datetime.utcnow()}
    
    return app

app = create_app()
celery = create_celery_app(app)

# Import models to ensure they are registered with SQLAlchemy
from app import models