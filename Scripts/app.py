# import the necessary packages
import eventlet
eventlet.monkey_patch()
from datetime import datetime
import numpy as np
from imagezmq import imagezmq
from imutils import resize
import argparse
import cv2
import requests
import time
import base64
from flask import Flask
from flask_socketio import SocketIO
from darkflow.net.build import TFNet
import os
from os.path import isfile, isdir, join
import uuid

print('[INFO] [{}] Load neural net (GPU)...'.format(str(datetime.now())))
options = {
    'model': 'cfg/yolov2.cfg',
    'load': 'weights/yolov2.weights',
    'threshold': 0.40,
    'gpu': 0.5
}
tfnet = TFNet(options)

# predict only every N milli seconds, so we
# are not killing the GPU + it removes some unnecessary flickering
PREDICT_TIMEOUT_MS = 150

# determine if images should be stored
STORE_IMAGES = True

# motion detection parameters
BG_SUB_HISTORY = 20
BG_SUB_THRESH = 30
BG_SUB_SHADOWS = True
MIN_OBJ_AREA = 35

# user friendly mapping for devices
DEVICE_MAP = {
    'lab-node2': 'Inside-Main-Door',
    'lab-node3': 'Front-Parking'
}

# initialize the consider set (class labels we care about and want
# to count), the object count dictionary, and the frame  dictionary
CONSIDER = set(["dog", "person", "car", "cat", "bird", "bicycle", "motorbike", "truck"])
streaming_devices = []

# create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# create socket server
socket = SocketIO(app, cors_allowed_origins='*', logger=False, engineio_logger=False)

# create function to emit a socket event 
# with data to the connected clients
def bg_emit(evt, data):
    socket.emit(evt, data)

# make sure that confidence is a float, 
# not numpy.float, which is not serializable
def norm_pred(pred):
    pred['confidence'] = pred['confidence'].astype(float)    
    return pred

# fire away a NN request to detect objects in the frame
def predict(frame):
    try:
        #start_time = datetime.now()
        preds = map(norm_pred, tfnet.return_predict(frame))
        #time_elapsed = (datetime.now() - start_time).microseconds / 1000
        #print('[INFO] [{}] Executed NN in {}ms'.format(str(datetime.now()), time_elapsed))
    except:
        print("[ERR] [{}] Unable to retrieve predictions from the model".format(str(datetime.now())))
        return []

    # remove any objects which we do not wish to track
    found_considered = [o for o in preds if o['label'] in CONSIDER]

    # further refine predictions:
    # Remove any cars parked right outside of the house,
    # the assumption here is that the bottomright y property will
    # be greater than 120 (for now)
    # TODO: move this to a separate mapping later
    found_refined = []
    for o in found_considered:
        if o['label'] == 'car' and o['bottomright']['y'] > 120:
            continue
        found_refined.append(o)
    if len(found_refined) > 0:
        print('[INFO] [{}] Found {} objects (refined count)'.format(str(datetime.now()), len(found_refined)))

    # return correct predictions
    preds = [] if len(found_refined) == 0 else found_refined
    return (preds, found_refined)

# save images on SSD
def store_image(frame, preds, device, objects_found):

    if len(objects_found) == 0:
        return None

    # figure out paths
    fpath = '/mnt/my-ssd/security_cam_detections_v2'
    date_now = datetime.now().strftime('%Y-%m-%d')
    
    # create directories if don't exist
    if isdir("{}/{}".format(fpath, device)) == False:
        print('[INFO] [{}] Creating dir {}/{}...'.format(str(datetime.now()), fpath, device))
        os.mkdir("{}/{}".format(fpath, device))
    if isdir("{}/{}/{}".format(fpath, device, date_now)) == False:
        print('[INFO] [{}] Creating dir {}/{}/{}...'.format(str(datetime.now()), fpath, device, date_now))
        os.mkdir("{}/{}/{}".format(fpath, device, date_now))

    # save file
    time_now = datetime.now().strftime('%H.%M.%S.%f')[:-3] # add milliseconds
    unique_id = str(uuid.uuid4())[:8]
    labels_found = "-".join([o['label'] for o in objects_found])
    fname = '{}_{}_{}.jpg'.format(time_now, unique_id, labels_found)

    full_fname = '{}/{}/{}/{}'.format(fpath, device, date_now, fname)
    print('[INFO] [{}] Saving file {}...'.format(str(datetime.now()), full_fname))

    return cv2.imwrite(full_fname, frame)

# create function to continously read data from the message queue
# and stream images to connected clients
def loop_all_the_time():
    
    # initialize the ImageHub object to collect images
    print('[INFO] [{}] Running ImageHub on tcp://*:5555'.format(str(datetime.now())))
    imageHub = imagezmq.ImageHub(open_port='tcp://*:5555')

    # create an instance of background subtractor,
    # which will be used to detect motion in frames
    subtractors = {}

    # lambda to get the current time in ms
    current_milli_time = lambda: int(round(time.time() * 1000))

    # initialise empty predictions and last api call timestamp,
    # this needs to be a dict as it is individual
    # for each streaming device
    preds = {}
    last_api_call_ts = {}
    recalibrate = {}
    stream_res = {}

    # create infinite loop and wait for images
    while True:

        # receive RPi name and frame from the RPi (using Message Queue)
        (rpi_name, jpg_buffer) = imageHub.recv_jpg()

        # use OpenCV and NumPy to decompress jpg
        frame = cv2.imdecode(np.fromstring(jpg_buffer, dtype='uint8'), -1)
        
        # ACK message
        imageHub.send_reply(b'OK')

        # map device name into a user friendly one
        rpi_name = DEVICE_MAP[rpi_name]

        # resize image for preprocessing and predictions
        frame_sm = resize(frame.copy(), width=608, height=608)

        # if a new device started streaming, print it, add to the list
        # and get initial predictions
        if rpi_name not in streaming_devices:
            print("[INFO] [{}] device started sending data: {}...".format(str(datetime.now()), rpi_name))
            streaming_devices.append(rpi_name)
            (preds[rpi_name], labels_found) = predict(frame_sm)
            if STORE_IMAGES == True:
                store_image(frame, preds[rpi_name], rpi_name, labels_found)
            last_api_call_ts[rpi_name] = current_milli_time()
            recalibrate[rpi_name] = False
            stream_res[rpi_name] = frame.shape[:2]
            subtractors[rpi_name] = cv2.createBackgroundSubtractorMOG2(history=BG_SUB_HISTORY, 
                varThreshold=BG_SUB_THRESH, detectShadows=BG_SUB_SHADOWS)
            
        # perform background subtraction to detect motion
        mask = subtractors[rpi_name].apply(frame_sm)

        # if change in frame has reached the threshold,
        # then it means there is some movement activity
        detect_objects = False
        
        # now we can find objects with an area greater than the 
        # treshold to filter out the noise from the frame
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) >= MIN_OBJ_AREA:
                detect_objects = True
                break
        
        # query API for the predictions if necessary,
        # we will also control the amount of API calls
        # we are making in 1 second to keep the screen smooth
        milli_time = current_milli_time()
        if detect_objects == True and milli_time - last_api_call_ts[rpi_name] > PREDICT_TIMEOUT_MS:
            (preds[rpi_name], labels_found) = predict(frame_sm)
            if STORE_IMAGES == True:
                store_image(frame, preds[rpi_name], rpi_name, labels_found)
            last_api_call_ts[rpi_name] = milli_time
            recalibrate[rpi_name] = True
        elif recalibrate[rpi_name] == True and milli_time - last_api_call_ts[rpi_name] > 1000:
            # after 1 second from last prediction,
            # run a prediction again to recalibrate the frame
            (preds[rpi_name], labels_found) = predict(frame_sm)
            if STORE_IMAGES == True:
                store_image(frame, preds[rpi_name], rpi_name, labels_found)
            recalibrate[rpi_name] = False
            
        # decode frame as base64, this seems to be much easier
        # in Node, and in Node clients can receive a clean base64
        # string. With Python I could not find a way to do this
        # and clients will need to decode the frame shipped as an 
        # ArrayBuffer from Python
        retval, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)

        # emit event to the clients
        bg_emit('IMAGE', {
            'device': rpi_name, 
            'img': jpg_as_text, 
            'boxes': preds[rpi_name],
            'res': stream_res[rpi_name]
        })
        
        # this is important so the program can release the thread 
        # (or something like that), without this the loop is stalled
        eventlet.sleep(0)
        
# create a background process
eventlet.spawn(loop_all_the_time)

# kick off Flask app
if __name__ == '__main__':
    socket.run(app, host='0.0.0.0', port=5001)