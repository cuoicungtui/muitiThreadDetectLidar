import cv2
import numpy as np
import time
from threading import Thread
import multiprocessing
import queue
# Import packages
import os
import argparse
import sys
import importlib.util
from ssdDetect import polygon_calculate
import threading

# Create a tracker based on tracker name
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
def createTrackerByName(trackerType):

    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
    return tracker



def detect_camera(videostream,imW,imH, result_queue,camera_thread_event):
    # ... (your existing code for camera detection)
    # Assuming PointsInfor is the result from camera detection
    MODEL_NAME = ''
    GRAPH_NAME = ""
    LABELMAP_NAME = ""
    min_conf_threshold = float(0.5)
    JSON_PATH = 'polygon.json'

    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    #Get path to current working directory
    CWD_PATH = os.getcwd()
    # path json polygon
    JSON_PATH = os.path.join(CWD_PATH,JSON_PATH)
    print("JSON path : ",JSON_PATH)

    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if labels[0] == '???':
        del(labels[0])
   
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5
    limit_area = 7000

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2
    
    # Get Polygon_calculate
    polygon_cal = polygon_calculate(JSON_PATH,imW,imH)
    # detect frame return boxes
    def detect_ssd(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

        boxes_new = []
        classes_new = []
        scores_new = []
        centroid_new = []
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                
                # scale boxes - values (0,1) to size witgh hight
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                if(polygon_cal.area_box((xmin,ymin,xmax,ymax),limit_area)):
                    centroid_new.append([int((xmin+xmax)//2),int((ymin+ymax)//2)])
                    boxes_new.append((xmin,ymin,xmax,ymax))
                    classes_new.append(classes[i])
                    scores_new.append(scores[i])
        return boxes_new,classes_new,scores_new,centroid_new

    boxes, classes,scores ,centroids_old = [],[],[],[]

    trackerType = trackerTypes[4]  
    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()
    count = 0
    num_frame_to_detect = 5

    while(True):
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = videostream.read()

        _, frame = polygon_cal.cut_frame_polygon(frame)

        # get updated location of objects in subsequent frames
        success, boxes_update = multiTracker.update(frame)

        if count == num_frame_to_detect:
            controids = polygon_cal.centroid(boxes_update)
            PointsInfor = polygon_cal.check_result(controids,centroids_old,frame)
            # print(f"Information point:{PointsInfor}  \n")
            result_queue.put(PointsInfor)
            count = 0
            camera_thread_event.set()

        if count == 0:
            boxes, classes,scores,centroids_old = detect_ssd(frame)
            multiTracker = cv2.MultiTracker_create()
            # Initialize MultiTracker
            for bbox in boxes:
                box_track = (bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])
                multiTracker.add(createTrackerByName(trackerType), frame, box_track)
    
            if len(scores)==0:
                count=-1
        count+=1


def detect_lidar(result_queue,lidar_thread_event):
   
    while True:
        cv2.waitKey(25)
        LidarData = {"datalida":"Lidar"}  # Replace this line with the actual result
        result_queue.put(LidarData)
        lidar_thread_event.set()

class VideoStream:
    """Camera object that controls video streaming"""
    def __init__(self,resolution=(640,480),framerate=30,STREAM_URL=''):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(STREAM_URL)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


def main_process():

    # Open video file
    VIDEO_PATH = 'rtsp://admin2:Atlab123@@192.168.1.64:554/Streaming/Channels/101'

    imW,imH = 1280,720
    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=25,STREAM_URL= VIDEO_PATH).start()
    time.sleep(1)

    # Initialize Queues to pass data between threads
    camera_result_queue = queue.Queue()
    lidar_result_queue = queue.Queue()

    camera_thread_event = threading.Event()
    lidar_thread_event = threading.Event()

    # Create and start the camera detection thread
    camera_thread = Thread(target=detect_camera, args=(videostream,imW,imH, camera_result_queue,camera_thread_event))
    camera_thread.start()

    # Create and start the lidar detection thread
    lidar_thread = Thread(target=detect_lidar, args=(lidar_result_queue,lidar_thread_event))
    lidar_thread.start()

    # Counter to track the number of received results
    results_received_count = 0

    while True:
        try:
            # Retrieve camera detection result from the Queue
            camera_result = camera_result_queue.get()
            print(f"Camera Detection Result: {camera_result}\n")
            
            # Process the camera_result as needed

            # Increment the counter
            results_received_count += 1
        except queue.Empty:
            pass

        try:
            # Retrieve lidar detection result from the Queue
            lidar_result = lidar_result_queue.get()
            print(f"Lidar Detection Result: {lidar_result}\n")
            
            # Process the lidar_result as needed

            # Increment the counter
            results_received_count += 1
        except queue.Empty:
            pass

        # Check if both camera and lidar results have been received
        if results_received_count >= 2:
            # Toggle the camera thread event to be used again
            camera_thread_event.clear()
            lidar_thread_event.clear()
            results_received_count = 0  # Reset the counter

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Terminate the camera and lidar threads
    camera_thread.join(timeout=0.1)
    camera_thread.terminate()

    lidar_thread.join(timeout=0.1)
    lidar_thread.terminate()

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()

if __name__ == "__main__":
    # Run the main process
    main_process()
