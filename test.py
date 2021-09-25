import cv2
import pyzbar.pyzbar as pyzbar

used_codes = []
data_list = []

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()

    for code in pyzbar.decode(frame):
        # cv2.imwrite('qrbarcode_image.png', frame)
        my_code = code.data.decode('utf-8')
        if my_code:
            print("인식 성공 : ", my_code)
        else:
            pass

    cv2.imshow('QRcode Barcode Scan', frame)

    cv2.waitKey(1)

