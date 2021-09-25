from flask import Flask, render_template, request, jsonify, Blueprint
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import rec
from pyzbar.pyzbar import decode
import pyzbar.pyzbar as pyzbar
import json
import time
from datetime import datetime, date
from flask_login import current_user

from .models import *
from . import db
from sqlalchemy import cast, DATE

# from pymongo import MongoClient

# client = MongoClient('localhost', 27017)
# db = client.hanium

home = Blueprint('home', __name__)

configuration_path = "./cfg/yolov3.cfg"
weights_path = "./yolov3.weights"

labels = open("./data/coco.names").read().strip().split('\n')

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for non maximum suppression
threshold = 0.3
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)

# Getting names of all layers
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]  # list of layers' names

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def html(content):
   return '<html><head></head><body>' + content + '</body></html>'

# 날짜별 기록
@home.route('/record')
def record():
    # user = User.query.get(current_user.id)
    # print(user)

    # today = datetime.today()
    # print(today)

    record_list = Record.query.filter_by(user_id=current_user.id) # 동일한 유저
    record_list.filter(cast(Record.date, DATE)==date.today()).all() # 같은 날짜
    print(record_list)

    food_list = []
    for record in record_list:
        # print(record.id, '\n\n')
        food = Food.query.filter_by(record_id=record.id)
        food_list += food

    print("\n\nfood_list:", food_list)
    return render_template('user/record.html', food_list=food_list)

# 상세 기록 정보
@home.route('/food_record')
def food_record():
    with open('./static/nutrition.json') as f:
        nutrition_data = json.load(f)
        print(json.dumps(nutrition_data))

    new_record = Record(user_id=current_user.id, date=datetime.now())
    db.session.add(new_record)
    db.session.commit()
    for food in products:
        for i in nutrition_data['nutrition']:
            if i['name'] == food:
                new_food = Food(record_id=new_record.id,
                                name=i['name'],
                                calories=i['calories'], 
                                sodium=i['sodium'], 
                                carbohydrate=i['carbohydrate'], 
                                fat=i['fat'], 
                                cholesterol=i['cholesterol'],
                                protein=i['protein'])
                db.session.add(new_food)
                db.session.commit()

    food_list = Food.query.filter_by(record_id = new_record.id).all()
    print(food_list) # <Food 29>, <Food 30>

    food_total = {}
    food_total['calories'] = food_total['sodium'] = food_total['carbohydrate'] \
    = food_total['fat'] = food_total['cholesterol'] = food_total['protein'] = 0
    for food in food_list:
        print("\n", food.calories)
        food_total['calories'] += food.calories
        food_total['sodium'] += food.sodium
        food_total['carbohydrate'] += food.carbohydrate
        food_total['fat'] += food.fat
        food_total['cholesterol'] += food.cholesterol
        food_total['protein'] += food.protein

    # print(food_total)
    # {'calories': 74.0, 'sodium': 102.0, 'carbohydrate': 17.0, 'fat': 0.4, 'cholesterol': 0.0, 'protein': 3.6999999999999997}

    return render_template('user/food_record.html', food_list=food_list, food_total=food_total)

#     foods = list(db.person.find({},{'_id':False}))
    
#     total_calories = 0
#     total_sodium = 0
#     total_carbohydrate = 0
#     total_fat = 0
#     total_cholesterol = 0
#     total_protein = 0
    
#     food_list = []
#     for food in foods:
#         food_name = food['name']
#         # food = db.food.find_one({'name': food_name})
#         food_list.append(food)

#         # 전체 계산
#         total_calories += food['calories']
#         total_sodium += food['sodium']
#         total_carbohydrate += food['carbohydrate']
#         total_fat += food['fat']
#         total_cholesterol += food['cholesterol']
#         total_protein += food['protein']
#         total_protein = round(total_protein, 3)
#     return render_template('home/record.html', food_list=food_list, total_calories=total_calories, total_sodium=total_sodium,
#                             total_carbohydrate=total_carbohydrate, total_fat=total_fat, total_cholesterol=total_cholesterol,
#                             total_protein=total_protein)

@home.route('/barcode', methods=['GET'])
def barcode():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, frame = cap.read()

        for code in pyzbar.decode(frame):
            my_code = code.data.decode('utf-8')
            if my_code:
                print("인식 성공 : ", my_code)
                cv2.destroyAllWindows()
                return render_template('home/barcode.html', my_code=my_code)
            else:
                pass
        cv2.imshow('Testing-code-scan', frame)
        cv2.waitKey(1)

@home.route('/barcode_record')
def barcode_record(my_code):
    pass

    # used_codes = []

    # camera = True
    # while camera == True:
        # success, frame = cap.read()

        # for code in decode(frame):

        #     if code.data.decode('utf-8') not in used_codes:
        #         print('Approved')
        #         barcode_num = code.data.decode('utf-8')
        #         print(barcode_num)
        #         used_codes.append(barcode_num)
        #         time.sleep(5)

        #         product = db.food.find_one({'barcode': barcode_num})
        #         print(product, '!!!!!!!!!!')
        #         db.person.insert_one(product)

        #         name = product['name']
        #         calories = product['calories']
        #         sodium = product['sodium']
        #         carbohydrate = product['carbohydrate']
        #         fat = product['fat']
        #         cholesterol = product['cholesterol']
        #         protein = product['protein']

        #         return render_template('home/barcode.html', name=name, calories=calories, sodium=sodium,
        #                                 carbohydrate=carbohydrate, fat=fat, cholesterol=cholesterol, protein=protein)

        #     elif code.data.decode('utf-8') in used_codes:
        #         print('Sorry, this code has been already used')
        #         time.sleep(5)
        #     else:
        #         return html('일치하는 바코드 번호가 없습니다. <a href="/home">메인페이지로 돌아가기</a>')

        # cv2.imshow('Testing-code-scan', frame)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()

@home.route('/')
def intro():
    return render_template('home/intro.html')

@home.route('/main')
def main():
    return render_template('home/main.html')

# @home.route('/camera')
# def camera():
#     return render_template('home/camera.html')

@home.route("/upload", methods=['GET', 'POST'])
def upload():
    return render_template('home/upload.html')


@home.route("/predict", methods = ['GET','POST'])
def predict():
    global products
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('./static/images/input', filename)
                file.save(file_path)
                pathImage = file_path
                image_input = img = cv2.imread(pathImage)
                image_input_shape = image_input.shape
      
                blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                # Slicing blob and transposing to make channels come at the end
                blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)           
                
                network.setInput(blob)  # setting blob as input to the network
                start = time.time()
                output_from_network = network.forward(layers_names_output)
                end = time.time()
                
                # Showing spent time for forward pass
                print('YOLO v3 took {:.5f} seconds'.format(end - start))
                np.random.seed(42)
                # randint(low, high=None, size=None, dtype='l')
                colors = np.random.randint(0, 255, size=(len(labels), 3))
                
                bounding_boxes = []
                confidences = []
                class_numbers = []
                h, w = image_input_shape[:2]  # Slicing from tuple only first two elements
                
                for result in output_from_network:
                    # Going through all detections from current output layer
                    for detection in result:
                        # Getting class for current object
                        scores = detection[5:]
                        class_current = np.argmax(scores)
                        confidence_current = scores[class_current]

                        if confidence_current > probability_minimum:    
                            box_current = detection[0:4] * np.array([w, h, w, h])
                            x_center, y_center, box_width, box_height = box_current.astype('int')
                            x_min = int(x_center - (box_width / 2))
                            y_min = int(y_center - (box_height / 2))
                
                            # Adding results into prepared lists
                            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                            confidences.append(float(confidence_current))
                            class_numbers.append(class_current)
                results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
            
                objects=[]
                for i in range(len(class_numbers)):
                    print(labels[int(class_numbers[i])])
                    objects.append(labels[int(class_numbers[i])])
                # Saving found labels
                """with open('found_labels.txt', 'w') as f:
                    for i in range(len(class_numbers)):
                        f.write(labels[int(class_numbers[i])])"""
                if len(results) > 0:
                    # Going through indexes of results
                    for i in results.flatten():
                        # Getting current bounding box coordinates
                        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                
                        colour_box_current = [int(j) for j in colors[class_numbers[i]]]
                
                        cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height),
                                        colour_box_current, 5)
                
                        
                        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                
                        cv2.putText(image_input, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, colour_box_current, 5)   
                cv2.imwrite('./static/images/output/' + filename , image_input)

                products = list(set(objects))
                print(products)
                    
                return render_template('home/predict.html', products = list(set(objects)), user_image = 'images/output/' + filename)
                
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."