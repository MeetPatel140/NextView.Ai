import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect
from celery import Celery
from config import get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()
celery = Celery("nextview")

def create_celery_app(app=None):
    app = app or create_app()
    celery.conf.update(app.config)
    
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                try:
                    return self.run(*args, **kwargs)
                except Exception as e:
                    logger.exception(f"Task {self.name} failed: {str(e)}")
                    raise
    
    celery.Task = ContextTask
    logger.info("Celery configured with broker: %s", celery.conf.broker_url)
    
    return celery

def create_app():
    app = Flask(__name__)
    app.config.from_object(get_config())
    
    # Configure Flask app logging
    if not app.debug:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
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
    create_celery_app(app)
    
    # Add context processor for template variables
    @app.context_processor
    def inject_now():
        from datetime import datetime
        return {'now': datetime.utcnow()}
    
    return app

# Create global celery instance
app = create_app()

# Import models to ensure they are registered with SQLAlchemy
from app import models

# Import tasks to ensure they are registered with Celery
with app.app_context():
    import app.main.routes