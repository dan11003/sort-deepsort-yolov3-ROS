#!/usr/bin/python3
import rospy
import numpy as np
from darknet_ros_msgs.msg import BoundingBoxes
from sort import sort 
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from sort_track.msg import IntList
from message_filters import TimeSynchronizer, Subscriber
import vision_msgs
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import *
from sensor_msgs.msg import CompressedImage



def get_parameters():
    """
    Gets the necessary parameters from .yaml file
    Returns tuple
    """
    camera_topic = rospy.get_param("~camera_topic")
    detection_topic = rospy.get_param("~detection_topic")
    tracker_topic = rospy.get_param('~tracker_topic')
    cost_threhold = rospy.get_param('~cost_threhold')
    min_hits = rospy.get_param('~min_hits')
    max_age = rospy.get_param('~max_age')
    return (camera_topic, detection_topic, tracker_topic, cost_threhold, max_age, min_hits)



"""
def callback_image(data):
    print("callback_image begin")
    #Display Image
    #bridge = CvBridge()
    #cv_rgb = bridge.imgmsg_to_cv2(data, "bgr8")
    np_arr = np.fromstring(data.data, np.uint8)
    cv_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #TO DO: FIND BETTER AND MORE ACCURATE WAY TO SHOW BOUNDING BOXES!!
    #Detection bounding box

    #print(len(detections))
    #print(detections)
    if(len(detections) > 0):
        for i in range(detections.shape[0]):
            #print("idx: " + str(i))
            print(detections[i][0])
            print(detections[i][1])
            print(detections[i][2])
            print(detections[i][3])
            print(detections[i][4])


            cv2.rectangle(cv_rgb, (int(detections[i][0]), int(detections[i][1])), (int(detections[i][2]), int(detections[i][3])), (100, 255, 50), 1)
            cv2.putText(cv_rgb , "person", (int(detections[i][0]), int(detections[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 50), lineType=cv2.LINE_AA)
    #Tracker bounding box
    #cv2.rectangle(cv_rgb, (track[0][0], self.track[0][1]), (track[0][2], self.track[0][3]), (255, 255, 255), 1)

    #cv2.putText(cv_rgb , str(track[0][4]), (track[0][2], self.track[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.imshow("YOLO+SORT", cv_rgb)
    cv2.waitKey(3)
    print("callback_image end")


def Detections(data):
    print("callbackVisionmsgDetection2dArray begin")
    #print(len(data.detections))
    #detections = np.array((0,5),dtype=numpy.float)
    global detections
    global self.trackers
    global self.track
    detections = []
    self.trackers = []
    self.track = []
    detectionsAggregate = []
    for detection in data.detections:
        bbox = detection.bbox
        confidence = round(float(detection.results[0].score),2)
        detectionsAggregate.append(np.array([int(bbox.center.x-bbox.size_x), int(bbox.center.y-bbox.size_y), int(bbox.center.x+bbox.size_x), int(bbox.center.y+bbox.size_y), confidence]))
    #print("got: " +str(len(detectionsAggregate)))



    if(len(detectionsAggregate) > 0):
        detections = np.array(detectionsAggregate)
        self.trackers = self.tracker.update(detections)
        self.trackers = np.array(trackers, dtype='int')
        self.track = self.trackers
        msg.data = self.track
    else:
        detections = np.array([])
    print("callbackVisionmsgDetection2dArray end")
    """


"""def callback_det(data):
    print("callback")
    print(len(data.detections))
    global detections
    global self.trackers
    global self.track
    detections = []
    self.trackers = []
    self.track = []
    for box in data.bounding_boxes:
        detections.append(np.array([box.xmin, box.ymin, box.xmax, box.ymax, round(box.probability,2)]))

    detections = np.array(detections)
    self.trackers = self.tracker.update(detections)
    self.trackers = np.array(trackers, dtype='int')
    self.track = self.trackers
    msg.data = self.track
"""

class MultiObjectTracker:

    def __init__(self):

        (camera_topic, detection_topic, tracker_topic, cost_threshold, max_age, min_hits) = get_parameters()
        self.tracker = sort.Sort(max_age=max_age, min_hits=min_hits) #create instance of the SORT self.tracker
        self.cost_threshold = cost_threshold

        self.track = np.array([])

        self.detect_sub = Subscriber("/detectnet/detections", Detection2DArray)
        self.image_sub  = Subscriber("/camera2/color/image_raw/compressed", CompressedImage)
        self.tss = TimeSynchronizer([self.detect_sub,self.image_sub], 100)
        self.tss.registerCallback(self.SyncedCallback)

        self.pub_trackers = rospy.Publisher(tracker_topic, IntList, queue_size=10)
        self.msg = IntList()




    def SyncedCallback(self,detectMsg, imageMsg):

        listDetections = []
        for detection in detectMsg.detections:
            bbox = detection.bbox
            confidence = round(float(detection.results[0].score),2)
            listDetections.append(np.array([int(bbox.center.x-bbox.size_x), int(bbox.center.y-bbox.size_y), int(bbox.center.x+bbox.size_x), int(bbox.center.y+bbox.size_y), confidence]))
        detections = np.array(listDetections)

        if(len(detections) > 0):
            listTrackers = self.tracker.update(detections)
            arrayTrackers = np.array(listTrackers, dtype='int')
            print("arrayTrackers")
            print(arrayTrackers.shape)
            self.track = arrayTrackers
            self.msg.data = self.track

        print("track")
        print(self.track)



        np_arr = np.fromstring(imageMsg.data, np.uint8)
        cv_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


                    #TO DO: FIND BETTER AND MORE ACCURATE WAY TO SHOW BOUNDING BOXES!!
        for i in range(detections.shape[0]):
            cv2.rectangle(cv_rgb, (int(detections[i][0]), int(detections[i][1])), (int(detections[i][2]), int(detections[i][3])), (100, 255, 50), 1)
            cv2.putText(cv_rgb , "person", (int(detections[i][0]), int(detections[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 50), lineType=cv2.LINE_AA)
        #Tracker bounding box
        for i in range(self.track.shape[0]):
            cv2.rectangle(cv_rgb, (self.track[i][0], self.track[i][1]), (self.track[i][2], self.track[i][3]), (255, 255, 255), 1)
            cv2.putText(cv_rgb , str(self.track[i][4]), (self.track[i][2], self.track[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow("YOLO+SORT", cv_rgb)
        cv2.waitKey(3)
        print("callback_image end")

        cv2.imshow("YOLO+SORT", cv_rgb)
        cv2.waitKey(3)
        #print(msg) #Testing msg that is published
        self.pub_trackers.publish(self.msg)



def main():
    print("Initialize ROS node")
    rospy.init_node('sort_tracker', anonymous=False)
    rate = rospy.Rate(10)
    rosTracker = MultiObjectTracker()
    while not rospy.is_shutdown():
        rate.sleep()
        rospy.spin()

    print("Finish self.tracking node")


if __name__ == '__main__':
    try :
        main()
    except rospy.ROSInterruptException:
        pass
