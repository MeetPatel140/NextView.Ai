from flask import Blueprint

bp = Blueprint('chatbot', __name__)

# Import routes at the bottom to avoid circular imports
from app.chatbot import routes