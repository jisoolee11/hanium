from flask import Blueprint, render_template
from . import db
from flask_login import login_required, current_user

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('auth/index.html')

@main.route('/profile')
@login_required
def profile():
    return render_template('auth/profile.html', name=current_user.name)