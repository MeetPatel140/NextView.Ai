from flask import Blueprint

bp = Blueprint('api', __name__)

# Import and register data visualization blueprint
from app.api.data_visualization import bp as data_viz_bp
bp.register_blueprint(data_viz_bp, url_prefix='/data_viz')

# Import routes at the bottom to avoid circular imports
from app.api import routes