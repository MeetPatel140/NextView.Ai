from datetime import datetime
from app import db
import json

class Dataset(db.Model):
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    file_path = db.Column(db.String(255), nullable=False)  # Path to the uploaded file
    file_type = db.Column(db.String(20), nullable=False)  # e.g., 'xlsx'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Processing metadata
    is_processed = db.Column(db.Boolean, default=False)
    row_count = db.Column(db.Integer, nullable=True)
    column_count = db.Column(db.Integer, nullable=True)
    dataset_metadata = db.Column(db.JSON, nullable=True)  # Store column info, data types, etc.
    
    # Relationships
    owner = db.relationship('User', back_populates='datasets', uselist=False)
    visualizations = db.relationship('Visualization', backref='dataset', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Dataset {self.name}>'    

class Visualization(db.Model):
    __tablename__ = 'visualizations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=True)
    chart_type = db.Column(db.String(50), nullable=False)  # e.g., 'bar', 'pie', 'line'
    config = db.Column(db.JSON, nullable=False)  # Store chart configuration
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('visualizations', lazy='dynamic'))
    
    def __repr__(self):
        return f'<Visualization {self.name}>'