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
from deep_sort import preprocessing as prep



CLASSES = {
  1 : "Onion",
  2 : "Weed",
  3 : "unknown"
}
COLORS = {
  1 : (0,255,0),
  2: (0,0,255),
  3: (255,0,0)
}
class MultiObjectTracker:


    def get_parameters(self):
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
        self.vis_detections = rospy.get_param('~visualize_detections')
        self.vis_tracked_objects = rospy.get_param('~visualize_tracked_objects')
        return (camera_topic, detection_topic, tracker_topic, cost_threhold, max_age, min_hits)

    def __init__(self):

        (camera_topic, detection_topic, tracker_topic, cost_threshold, max_age, min_hits) = self.get_parameters()
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

        print("callback")
        #listDetections = []
        #listDetections = [list(np.array()) for i in range(CLASSES)]
        listDetections = dict()

        #for i in CLASSES.keys():
        #    listDetections[i] = list(np.array([]))

        listDetections[i] = [list(np.array([])) for i in CLASSES.keys()]
        print(listDetections)



        print(type(listDetections))
        print(listDetections)
        for detection in detectMsg.detections:
            bbox = detection.bbox
            label = int(detection.results[0].id)
            confidence = round(float(100*detection.results[0].score),2)
            print("at: " + str(label))
            npDetection = np.array([int(bbox.center.x-bbox.size_x), int(bbox.center.y-bbox.size_y), int(bbox.center.x+bbox.size_x), int(bbox.center.y+bbox.size_y), confidence, label])
            listDetections[label].append(npDetection)


        #scores_new = np.array([d.confidence for d in detections_new])
        #indices = prep.non_max_suppression(boxes, 1.0 , scores_new)
        #detections_new = [detections_new[i] for i in indices]
        detections = {}
        detections = np.array(listDetections)

        if(detections.shape[0] > 0):
            boxes = detections[:,0:4]#np.array([d.tlwh for d in detections])
            confidence = detections[:,4]

            indices = prep.non_max_suppression(boxes, 0.9 , confidence)
            detectionsFiltered = [detections[i] for i in indices]
            detections = np.array(detectionsFiltered)

        #print("detections filtered size: " + str(detections.shape[0]))

        if(len(detections) > 0):
            listTrackers = self.tracker.update(detections)
            arrayTrackers = np.array(listTrackers, dtype='int')
            self.track = arrayTrackers
            self.msg.data = self.track

        #print("track")
        #print(self.track)



        np_arr = np.fromstring(imageMsg.data, np.uint8)
        cv_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


                    #TO DO: FIND BETTER AND MORE ACCURATE WAY TO SHOW BOUNDING BOXES!!
        if self.vis_detections:
            for i in range(detections.shape[0]):
                cv2.rectangle(cv_rgb, (int(detections[i][0]), int(detections[i][1])), (int(detections[i][2]), int(detections[i][3])), COLORS[detections[i][5]], 1)
                cv2.putText(cv_rgb , CLASSES[detections[i][5]], (int(detections[i][0]), int(detections[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 50), lineType=cv2.LINE_AA)
        #Tracker bounding box
        if self.vis_tracked_objects:
            for i in range(self.track.shape[0]):
                cv2.rectangle(cv_rgb, (self.track[i][0], self.track[i][1]), (self.track[i][2], self.track[i][3]), (255, 255, 255), 1)
                cv2.putText(cv_rgb , str(self.track[i][4]), (self.track[i][2], self.track[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.imshow("YOLO+SORT", cv_rgb)
        cv2.waitKey(3)


        cv2.imshow("YOLO+SORT", cv_rgb)
        cv2.waitKey(3)
        #print(msg) #Testing msg that is published
        self.pub_trackers.publish(self.msg)



def main():
    print("Initialize ROS node")
    rospy.init_node('sort_tracker', anonymous=False)
    rate = rospy.Rate(10)
    rosTracker = {}
    for i in CLASSES.keys():
        rosTracker[i] = MultiObjectTracker()

    #print(type(rosTracker))
    #print(rosTracker)


    #rosTracker = MultiObjectTracker()
    print("Initialize finished")
    while not rospy.is_shutdown():
        rate.sleep()
        rospy.spin()

    print("Finish self.tracking node")


if __name__ == '__main__':
    try :
        main()
    except rospy.ROSInterruptException:
        pass
