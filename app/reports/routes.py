from flask import render_template, flash, redirect, url_for, request, current_app
from flask_login import login_required, current_user
from app.reports import bp

@bp.route('/dashboard')
@login_required
def dashboard():
    """Display the reports dashboard"""
    return render_template('reports/dashboard.html', title='Reports Dashboard')

@bp.route('/generate')
@login_required
def generate_report():
    """Generate a new report"""
    # This is a placeholder for report generation functionality
    flash('Report generation feature is coming soon!', 'info')
    return redirect(url_for('reports.dashboard'))