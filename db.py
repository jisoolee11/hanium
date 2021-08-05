from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.hanium


# ###################################################### barcode ######################################################
# db.food.insert_one(
#    { "barcode": "8801073210363", "name": "불닭볶음면", "calories": 425, "sodium": 950, "carbohydrate": 8,
#    "fat":15, "cholesterol":0, "protein":9}
# )

# db.food.insert_one(
#    { "barcode": "8800279", "name": "바나나맛우유", "calories": 208, "sodium": 110, "carbohydrate": 27,
#    "fat":8, "cholesterol":30, "protein":7}
# )

# db.food.insert_one(
#    { "barcode": "8809461790756", "name": "게이밍마우스패드", "calories": 208, "sodium": 110, "carbohydrate": 27,
#    "fat":8, "cholesterol":30, "protein":7}
# )

#########################################################################################################################

# ###################################################### nutrition ######################################################
# db.food.insert_one(
#    { "id": 0, "name": "banana", "calories": 89, "sodium": 1, "carbohydrate": 23,
#    "fat":0.3, "cholesterol":0, "protein":1.1}
# )

# db.food.insert_one(
#    { "id": 1, "name": "apple", "calories": 52, "sodium": 1, "carbohydrate": 14,
#    "fat":0.2, "cholesterol":0, "protein":0.3}
# )

# db.food.insert_one(
#    { "id": 2, "name": "orange", "calories": 44, "sodium": 1, "carbohydrate": 10,
#    "fat":0.2, "cholesterol":0, "protein":0.7}
# )

# db.food.insert_one(
#    { "id": 3, "name": "broccoli", "calories": 33, "sodium": 33, "carbohydrate": 7,
#    "fat":0.2, "cholesterol":0, "protein":2.8}
# )

# db.food.insert_one(
#    { "id": 4, "name": "carrot", "calories": 41, "sodium": 69, "carbohydrate": 10,
#    "fat":0.2, "cholesterol":0, "protein":0.9}
# )