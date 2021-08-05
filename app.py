from flask import Flask, render_template, request, jsonify, Blueprint
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from pyzbar.pyzbar import decode
import json
import time

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.hanium

app = Blueprint('app', __name__)

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

@app.route('/record')
def food_record():
    foods = list(db.person.find({},{'_id':False}))

    food_list = []
    for food in foods:
        food_name = food['name']
        food = db.food.find_one({'name': food_name})
        food_list.append(food)
    return render_template('home/record.html', food_list=food_list)

@app.route('/barcode', methods=['GET'])
def barcode():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    used_codes = []

    camera = True
    while camera == True:
        success, frame = cap.read()

        for code in decode(frame):

            if code.data.decode('utf-8') not in used_codes:
                print('Approved')
                barcode_num = code.data.decode('utf-8')
                print(barcode_num)
                used_codes.append(barcode_num)
                time.sleep(5)

                product = db.food.find_one({'barcode': barcode_num})
                print(product)
                name = product['name']
                calories = product['calories']
                sodium = product['sodium']
                carbohydrate = product['carbohydrate']
                fat = product['fat']
                cholesterol = product['cholesterol']
                protein = product['protein']

                db.person.insert_one(product)
                return render_template('home/barcode.html', name=name, calories=calories, sodium=sodium,
                                        carbohydrate=carbohydrate, fat=fat, cholesterol=cholesterol, protein=protein)

            elif code.data.decode('utf-8') in used_codes:
                print('Sorry, this code has been already used')
                time.sleep(5)
            else:
                return html('일치하는 바코드 번호가 없습니다. <a href="/home">메인페이지로 돌아가기</a>')

        cv2.imshow('Testing-code-scan', frame)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

@app.route('/home')
def index():
    return render_template('home/index.html')

@app.route('/camera')
def camera():
    return render_template('home/cam.html')

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    return render_template('home/upload.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
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
                for product in products:
                    food = db.food.find_one({'name': product})
                    db.person.insert_one(food)
                    
                return render_template('home/predict.html', products = products, user_image = 'images/output/' + filename)
                
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

'''
 <form action="http://127.0.0.1:5000/" onLoad="LoadOnce()">
      <input type="submit" value="wanna find other things ? ?" />
  </form>
'''