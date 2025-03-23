from flask import Blueprint

bp = Blueprint('reports', __name__)

# Import routes at the bottom to avoid circular imports
from app.reports import routes