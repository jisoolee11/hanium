from flask import Blueprint, render_template, request
from . import db
from flask_login import login_required, current_user
from .models import *

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

@user.route('/bmi/result', methods=['GET', 'POST'])
def bmi_result():
    bmi = ''
    if request.method == 'POST':
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height'))
        bmi = calc_bmi(weight, height)

        user = User.query.get(current_user.id)
        user.weight = weight
        user.height = height
        db.session.commit()

    return render_template("user/bmi_result.html", weight=weight, height=height, bmi=bmi)

def calc_bmi(weight, height):
    return round((weight / ((height / 100) ** 2)), 2)

