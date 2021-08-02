from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.hanium

# db.barcode.insert_one(
#    { "barcode": "880107321036", "name": "불닭볶음면", "calories(kcal)": 425, "sodium(mg)": 950, "carbohydrate(g)": 8,
#    "fat(g)":15, "cholesterol(mg)":0, "protein(g)":9}
# )

# db.barcode.insert_one(
#    { "barcode": "8800279", "name": "바나나맛우유", "calories(kcal)": 208, "sodium(mg)": 110, "carbohydrate(g)": 27,
#    "fat(g)":8, "cholesterol(mg)":30, "protein(g)":7}
# )

# db.barcode.insert_one(
#    { "barcode": "8809461790756", "name": "게이밍마우스패드", "calories(kcal)": 208, "sodium(mg)": 110, "carbohydrate(g)": 27,
#    "fat(g)":8, "cholesterol(mg)":30, "protein(g)":7}
# )