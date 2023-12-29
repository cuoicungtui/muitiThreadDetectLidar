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
import json


# Import lib Lidar
import serial
from CalcLidarData import CalcLidarData

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


def detect_camera(videostream,imW,imH, result_queue,camera_thread_event):
    # ... (your existing code for camera detection)
    # Assuming PointsInfor is the result from camera detection
    MODEL_NAME = './All_Model_detect/Sample_TFLite_model'
    GRAPH_NAME = "detect.tflite"
    LABELMAP_NAME = "labelmap.txt"
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
        # print("Start \n")
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = videostream.read()

        _, frame = polygon_cal.cut_frame_polygon(frame)

        # get updated location of objects in subsequent frames
        success, boxes_update = multiTracker.update(frame)

        if count == num_frame_to_detect:
            controids = polygon_cal.centroid(boxes_update)
            PointsInfor = polygon_cal.check_result(controids,centroids_old)
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
    
            # if len(scores)==0:
            #     count=-1
        count+=1
        # print(count)
        # cv2.imshow('Object detector 1', frame)



CHECK_CAM = False

def detect_lidar(ser,angle_min,angle_max,total_points,result_queue,lidar_thread_event):

    global CHECK_CAM

    tmpString = ""

    angle_old = 0
    list_lidar_point = list(range(total_points))
    Index_list = 0
    while True:
        loopFlag = True
        flag2c = False

        while loopFlag:
            b = ser.read()
            tmpInt = int.from_bytes(b, 'big')

            if tmpInt == 0x54:
                tmpString += b.hex() + " "
                flag2c = True
                continue

            elif tmpInt == 0x2c and flag2c:
                tmpString += b.hex()

                if not len(tmpString[0:-5].replace(' ', '')) == 90:
                    tmpString = ""
                    loopFlag = False
                    flag2c = False
                    continue

                lidarData = CalcLidarData(tmpString[0:-5])
                # print(f"lidar information: Distance{lidarData.Distance_i} angle : {lidarData.Angle_i} \n")
                # print(f"lidar information: Sum Distance{sum(lidarData.Distance_i)} len_distance {len(lidarData.Distance_i)} \n")
                # print(f"lidar information: start {lidarData.Angle_i[0]} end {lidarData.Angle_i[11]} len_distance {len(lidarData.Distance_i)} \n")
                # add point lidar
                
                
                for angle,distance in zip(lidarData.Angle_i,lidarData.Distance_i):
                    if  angle > angle_min and angle< angle_max:
                        if  angle > angle_old:
                            list_lidar_point[Index_list] = distance
                            Index_list+=1
                            angle_old = angle
                        else:

                            Index_list = 0
                            list_lidar_point[Index_list] = distance
                            Index_list+=1
                            angle_old = angle

                tmpString = ""
                loopFlag = False


            else:
                tmpString += b.hex() + " "

            flag2c = False
        if CHECK_CAM:
            CHECK_CAM = False
            LidarData = {"datalida":list_lidar_point}  # Replace this line with the actual result
            result_queue.put(LidarData)
            lidar_thread_event.set()


def result_led_right(): 
    print("Turn on right \n")


def result_led_left():
    print("Turn on left \n")



def main_process():

    INDEX_CHECK = 0
    NUM_Check_Lidar = 15
    CHECK_FRAME_LIDAR = np.zeros(NUM_Check_Lidar)
    CHECK_FRAME_LEFT = np.zeros(NUM_Check_Lidar)
    CHECK_FRAME_RIGHT = np.zeros(NUM_Check_Lidar)

    NUM_CHECK_WARNING = 20
    INDEX_WARNING= 0
    CHECK_FRAME_FORBIDDEN = np.zeros(NUM_CHECK_WARNING)
    CHECK_FRAME_FREEZE = np.zeros(NUM_CHECK_WARNING)


    global CHECK_CAM
    # Connect Camera IP
    VIDEO_PATH = 'rtsp://admin2:Atlab123@@192.168.1.64:554/Streaming/Channels/101'

    imW,imH = 1280,720
    videostream = VideoStream(resolution=(imW,imH),framerate=25,STREAM_URL= VIDEO_PATH).start()
    time.sleep(1)

    # Connect Lidar port
    ser = serial.Serial(port='/dev/ttyS0',
                    baudrate=230400,
                    timeout=5.0,
                    bytesize=8,
                    parity='N',
                    stopbits=1)

    Lidar_path_json = 'lidar_infor.json'
    with open(Lidar_path_json,'r') as json_file:
        Lidar_infor = json.load(json_file)

    angle_min = Lidar_infor['angle_min']
    angle_max = Lidar_infor['angle_max']
    total_points = Lidar_infor['Len_points']
    Sum_distance_max = Lidar_infor['Sum_distance']

    Mean_err = 2

    # Initialize Queues to pass data between threads
    camera_result_queue = queue.Queue()
    lidar_result_queue = queue.Queue()

    camera_thread_event = threading.Event()
    lidar_thread_event = threading.Event()

    # Create and start the camera detection thread
    camera_thread = Thread(target=detect_camera, args=(videostream,imW,imH, camera_result_queue,camera_thread_event))
    camera_thread.start()

    # Create and start the lidar detection thread
    lidar_thread = Thread(target=detect_lidar, args=( ser,angle_min,angle_max,total_points,lidar_result_queue,lidar_thread_event))
    lidar_thread.start()

    # Counter to track the number of received results
    results_received_count = 0
    All_result = {}
    
    index_count = 0
    while True:
        try:
            # Retrieve camera detection result from the Queue
            camera_result = camera_result_queue.get()
            All_result = {
                'Left':camera_result['Left'],
                'Right':camera_result['Right'],
                'Forbidden' : camera_result['Forbidden'],
                'freeze' : camera_result['freeze']
            }
            CHECK_CAM = True
            # Process the camera_result as needed

            # Increment the counter
            results_received_count += 1
        except camera_result_queue.Empty:
            pass

        try:
            # Retrieve lidar detection result from the Queue
            lidar_result = lidar_result_queue.get()
            #print(f"Lidar Detection Result: {len(lidar_result['datalida']) } sum_distance : {sum(lidar_result['datalida'])}\n")
            SUM_DISTANCE = sum(lidar_result['datalida'])

            if (abs(SUM_DISTANCE- Sum_distance_max) > (Mean_err* total_points)):

                All_result['Lidar_infor'] = True
            else:
                All_result['Lidar_infor'] = False
            # Increment the counter
            results_received_count += 1
        except lidar_result_queue.Empty:
            pass

        # Check if both camera and lidar results have been received
        if results_received_count >= 2:
            # Toggle the camera thread event to be used again
            # CONTROL LED

            if All_result['Lidar_infor']:
                CHECK_FRAME_LIDAR[INDEX_CHECK] = 1
            if All_result['Left']:
                CHECK_FRAME_LEFT[INDEX_CHECK] = 1
            if All_result['Right']:
                CHECK_FRAME_RIGHT[INDEX_CHECK] = 1
            if INDEX_CHECK == NUM_Check_Lidar -1:
                INDEX_CHECK = 0
            else:
                INDEX_CHECK+=1

            if All_result['Forbidden']:
                CHECK_FRAME_FORBIDDEN[INDEX_WARNING] = 1
            if All_result['freeze']:
                CHECK_FRAME_FREEZE[INDEX_WARNING] = 1

            if INDEX_WARNING == NUM_CHECK_WARNING -1:
                INDEX_WARNING = 0
            else:
                INDEX_WARNING+=1


            if sum(CHECK_FRAME_LEFT)/NUM_Check_Lidar >0.7 and  sum(CHECK_FRAME_LIDAR)/NUM_Check_Lidar >0.5  :
                print("CALL TURN ON LEFT")
            if sum(CHECK_FRAME_LEFT)/NUM_Check_Lidar <0.3  :
                print("CALL TURN OFF LEFT")


            if sum(CHECK_FRAME_RIGHT)/NUM_Check_Lidar >0.7 and  sum(CHECK_FRAME_LIDAR)/NUM_Check_Lidar >0.5  :
                print("CALL TURN ON RIGHT")
            if sum(CHECK_FRAME_RIGHT)/NUM_Check_Lidar <0.3  :
                print("CALL TURN OFF RIGHT")


            if (sum(CHECK_FRAME_FORBIDDEN)/NUM_CHECK_WARNING >0.7 or sum(CHECK_FRAME_FREEZE)/NUM_CHECK_WARNING >0.7 )and  sum(CHECK_FRAME_LIDAR)/NUM_Check_Lidar >0.5  :
                print("CALL TURN ON Warning")
            if sum(CHECK_FRAME_FORBIDDEN)/NUM_CHECK_WARNING <0.3  :
                print("CALL TURN OFF Warning")


            print(f"Infor : {All_result} {index_count} \n" )
            index_count+=1

            All_result.clear()
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
