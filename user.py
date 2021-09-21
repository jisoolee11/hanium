from flask import Blueprint, render_template
from . import db
from flask_login import login_required, current_user

user = Blueprint('user', __name__)

@user.route('/profile')
@login_required
def profile():
    return render_template('user/profile.html')

@user.route('/record')
def record():
    return render_template('user/record.html')

@user.route('/bmi')
def bmi():
    return render_template('user/bmi.html')