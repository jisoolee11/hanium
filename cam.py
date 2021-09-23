from flask import Flask, render_template, Response, Blueprint, request
import cv2
import datetime, time
import os

cam = Blueprint('cam', __name__)
 # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    camera = cv2.VideoCapture(0) 
    global frame
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

            


@cam.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@cam.route('/camera')
def camera():
    """Video streaming home page."""
    return render_template('home/cam.html')

@cam.route('/requests',methods=['POST','GET'])
def tasks():
    global p
    if request.method == 'POST':
        if request.form.get('capture'):
            now = datetime.datetime.now()
            p = os.path.sep.join(['static/shots', "shot_{}.jpg".format(str(now).replace(":",''))])
            cv2.imwrite(p, frame)

    return render_template('home/cam_image.html', image_path=p[7:].replace("\\", "/"))

@cam.route('/retry')
def retry():
    os.remove(p)
    return render_template('home/cam.html')
