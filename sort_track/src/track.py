#!/usr/bin/python3
import rospy
import numpy as np
#from darknet_ros_msgs.msg import BoundingBoxes
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
        self.camera_topic = rospy.get_param("~camera_topic")
        self.detection_topic = rospy.get_param("~detection_topic")
        self.tracker_topic = rospy.get_param('~tracker_topic')
        self.cost_threshold = rospy.get_param('~cost_threhold')
        self.min_hits = rospy.get_param('~min_hits')
        self.max_age = rospy.get_param('~max_age')
        self.IOU_th_association = rospy.get_param('~IOU_th_association')
        self.VerifyByDetectionRate = rospy.get_param('~VerifyByDetectionRate')
        self.vis_raw_detections = rospy.get_param('~visualize_raw_detections')
        self.vis_filtered_detections = rospy.get_param('~visualize_filtered_detections')
        self.vis_tracked_objects = rospy.get_param('~visualize_tracked_objects')
        self.vis_unreliable_objects = rospy.get_param('~visualize_unreliable_objects')
        self.non_max_suppression_th = rospy.get_param('~non_max_suppression_th')


    def __init__(self):

        self.get_parameters()
        self.tracker = {}
        for i in CLASSES.keys():
            self.tracker[i] = sort.Sort(max_age=self.max_age, min_hits=self.min_hits, IOU_th_association=self.IOU_th_association, VerifyByDetectionRate=self.VerifyByDetectionRate) #create instance of the SORT self.tracker


        self.track = np.array([])

        self.detect_sub = Subscriber("/detectnet/detections", Detection2DArray)
        self.image_sub  = Subscriber("/camera2/color/image_raw/compressed", CompressedImage)
        self.tss = TimeSynchronizer([self.detect_sub,self.image_sub], 100)
        self.tss.registerCallback(self.SyncedCallback)

        self.pub_trackers = rospy.Publisher(self.tracker_topic, IntList, queue_size=10)
        self.msg = IntList()
        self.frame = 0

    # this takes all detections, separates these per label, perform data processing and tracking
    def ProcessAndTrack(self,detectMsg):
        self.sortedDetections = {} # Raw detections separated per label before filtering. This will be used for mostly debugging the filtering step
        self.detectSortedFilt = {} # Detections after filtering separated per label. This is the input to the tracker. These detection must be "visually correct" with objects that require tracking
        trackedObjLoc = {} # output, nu
        unreliableLoc = {}
        self.frame = self.frame + 1

        listDetections = dict()
        for i in CLASSES.keys():
            listDetections[i] = list(np.array([]))

        for detection in detectMsg.detections:
            bbox = detection.bbox
            label = int(detection.results[0].id)
            confidence = round(float(100*detection.results[0].score),2)
            npDetection = np.array([int(bbox.center.x-bbox.size_x), int(bbox.center.y-bbox.size_y), int(bbox.center.x+bbox.size_x), int(bbox.center.y+bbox.size_y), confidence, label])
            listDetections[label].append(npDetection)


        for idxLbl in CLASSES.keys():
            self.sortedDetections[idxLbl] = np.array(listDetections[idxLbl])

        for idxLbl in CLASSES.keys():
            detectionsRaw = self.sortedDetections[idxLbl]

            if(detectionsRaw.shape[0] > 0):
                boxes = detectionsRaw[:,0:4]
                confidence = detectionsRaw[:,4]
                print("non_max_suppression_th: " + str(self.non_max_suppression_th))
                indices = prep.non_max_suppression(boxes, self.non_max_suppression_th , confidence)
                detectionsFilteredTmp = [detectionsRaw[i] for i in indices]
                self.detectSortedFilt[idxLbl] = np.array(detectionsFilteredTmp)
                #print("filtering from {0} to {1}".format(detectionsRaw.shape[0],self.detectSortedFilt[idxLbl].shape[0]))
                #print("filtering of " + str(idxLbl))
                #print("filtered from: " + str(detectionsRaw) +str(len(self.detectSortedFilt[idxLbl])))
                #print()
            else:
                #print("empty" + str(idxLbl))
                self.detectSortedFilt[idxLbl] = np.array([]) # empty
            #print("detections filtered size: " + str(detections.shape[0]))

            #if(len(self.detectSortedFilt[idxLbl]) > 0): #if has detections. why does it needs to have detections?? please try to remove this
            trackingInfo, count, unreliableTrackInfo = self.tracker[idxLbl].update(self.detectSortedFilt[idxLbl])
            trackedObjLoc[idxLbl] = np.array(trackingInfo, dtype='int')
            unreliableLoc[idxLbl] = np.array(unreliableTrackInfo, dtype='int')
                #print("update idx:  " +str(idxLbl) + str(trackedObjLoc[idxLbl]))
            #else:
            #    trackedObjLoc[idxLbl] = np.array([])

        return trackedObjLoc, unreliableLoc


    def SyncedCallback(self,detectMsg, imageMsg):

        #print("callback")
        trackedObject, trackedUnreliable = self.ProcessAndTrack(detectMsg)
        #Image processing
        np_arr = np.fromstring(imageMsg.data, np.uint8)
        cv_rgb = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


                    #TO DO: FIND BETTER AND MORE ACCURATE WAY TO SHOW BOUNDING BOXES!!
        strDisplay =[]
        if self.vis_tracked_objects:
            strDisplay.append("tracked_objects")
            for idxLbl in CLASSES.keys():
                track = trackedObject[idxLbl]
                for i in range(track.shape[0]):
                    cv2.rectangle(cv_rgb, (track[i][0], track[i][1]), (track[i][2], track[i][3]), COLORS[idxLbl], 4)
                    cv2.putText(cv_rgb , "Trk." + str(track[i][4]) + ",s: "+str(track[i][5]), (track[i][2], track[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idxLbl], lineType=cv2.LINE_AA)

        if self.vis_unreliable_objects:
            strDisplay.append("tracked_unreliable")
            for idxLbl in CLASSES.keys():
                track = trackedUnreliable[idxLbl]
                for i in range(track.shape[0]):
                    cv2.rectangle(cv_rgb, (track[i][0], track[i][1]), (track[i][2], track[i][3]), COLORS[idxLbl], 2)
                    cv2.putText(cv_rgb , "Unr." + str(track[i][4]), (track[i][2], track[i][3]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idxLbl], lineType=cv2.LINE_AA)

        if self.vis_filtered_detections:
            strDisplay.append("filtered objects")
            for idxLbl in CLASSES.keys():
                detections = self.detectSortedFilt[idxLbl]
                for i in range(detections.shape[0]):
                    cv2.rectangle(cv_rgb, (int(detections[i][0]), int(detections[i][1])), (int(detections[i][2]), int(detections[i][3])), COLORS[idxLbl], 1)
                    cv2.putText(cv_rgb , "Flt." + CLASSES[detections[i][5]], (int(detections[i][0]), int(detections[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idxLbl], lineType=cv2.LINE_AA)

        if self.vis_raw_detections:
            strDisplay.append("raw detections")
            for idxLbl in CLASSES.keys():
                detections = self.sortedDetections[idxLbl]
                for i in range(detections.shape[0]):
                    cv2.rectangle(cv_rgb, (int(detections[i][0]), int(detections[i][1])), (int(detections[i][2]), int(detections[i][3])), COLORS[idxLbl], 1)
                    cv2.putText(cv_rgb , "Raw." + CLASSES[detections[i][5]] +" " +str(detections[i][4]), (int(detections[i][2]), int( (detections[i][1]+detections[i][3])/2)  ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[idxLbl], lineType=cv2.LINE_AA)
        ObjectCountReport = []
        for i in CLASSES.keys():
            ObjectCountReport.append( "{0}: {1}".format(CLASSES[i],self.tracker[i].realObjectsCounter) )
        ObjectCountReport.append( "frame {0}".format(self.frame) )
        cv2.putText(cv_rgb , " ".join(ObjectCountReport), (18, cv_rgb.shape[1]-200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), lineType=cv2.LINE_AA)



        cv2.imshow(" ".join(strDisplay), cv_rgb)
        cv2.waitKey(3)

        #print(msg) #Testing msg that is published
        #self.pub_trackers.publish(self.msg)



def main():
    print("Initialize ROS node")
    rospy.init_node('sort_tracker', anonymous=False)
    rate = rospy.Rate(10)
    rosTracker = MultiObjectTracker()

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
