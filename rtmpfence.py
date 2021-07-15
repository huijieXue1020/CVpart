# -*- coding: utf-8 -*-

"""
禁止区域入侵检测

用法:


"""

import argparse
import os
import time
import cv2
import queue
from threading import Thread
import datetime, _thread
import subprocess as sp

from A_Final_Sys.oldcare.track import CentroidTracker
from A_Final_Sys.oldcare.track import TrackableObject
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2


VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ANGLE = 20
# 全局常量

limit_time = 2
# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()

# rtmpUrl = "rtmp://127.0.0.1:1935/rtmplive"
rtmpUrl = "rtmp://192.168.250.109:1935/rtmplive"

command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(640, 480),  # 图片分辨率
           '-r', str(20.0),  # 视频帧率
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           rtmpUrl]


def Video():
    # 调用相机拍图的函数

    vid = cv2.VideoCapture(0)
    time.sleep(2);
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    while (vid.isOpened()):
        return_value, frame = vid.read()

        # 原始图片推入队列中

        frame_queue.put(frame)
        frame_queue.get() if frame_queue.qsize() > 1 else time.sleep(0.01)



def push_frame():
    prototxt_file_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
    model_file_path = 'models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
    skip_frames = 30

    # 超参数
    # minimum probability to filter weak detections
    minimum_confidence = 0.80

    # 物体识别模型能识别的物体（21种）
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # start the frames per second throughput estimator
    # fps = FPS().start()



    # 推流函数


    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    # prev_time = time()
    # 防止多线程时 command 未被设置

    while True:
        print('command lenth', len(command))
        if len(command) > 0:
            # 管道配置，其中用到管道
            p = sp.Popen(command, stdin=sp.PIPE)
            break

    while True:
        # 加载物体识别模型
        net = cv2.dnn.readNetFromCaffe(prototxt_file_path, model_file_path)
        if frame_queue.empty() != True:
            counter = 0

            while True:
                counter += 1
                image = frame_queue.get()
                image = cv2.flip(image, 1)  #镜像翻转

                frame = imutils.resize(image, width=500)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # if the frame dimensions are empty, set them
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                # initialize the current status along with our list of bounding
                # box rectangles returned by either (1) our object detector or
                # (2) the correlation trackers
                status = "Waiting"
                rects = []

                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if totalFrames % skip_frames == 0:
                    # set the status and initialize our new set of object trackers
                    status = "Detecting"
                    trackers = []

                    # convert the frame to a blob and pass the blob through the
                    # network and obtain the detections
                    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                    net.setInput(blob)
                    detections = net.forward()

                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated
                        # with the prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by requiring a minimum
                        # confidence
                        if confidence > minimum_confidence:
                            # extract the index of the class label from the
                            # detections list
                            idx = int(detections[0, 0, i, 1])

                            # if the class label is not a person, ignore it
                            if CLASSES[idx] != "person":
                                continue

                            # compute the (x, y)-coordinates of the bounding box
                            # for the object
                            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                            (startX, startY, endX, endY) = box.astype("int")

                            # construct a dlib rectangle object from the bounding
                            # box coordinates and then start the dlib correlation
                            # tracker
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY)) ###change by wangao
                            tracker.start_track(rgb, rect)

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            trackers.append(tracker)

                # otherwise, we should utilize our object *trackers* rather than
                # object *detectors* to obtain a higher frame processing throughput
                else:
                    # loop over the trackers
                    for tracker in trackers:
                        # set the status of our system to be 'tracking' rather
                        # than 'waiting' or 'detecting'
                        status = "Tracking"

                        # update the tracker and grab the updated position
                        tracker.update(rgb)
                        pos = tracker.get_position()

                        # unpack the position object
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())

                        # draw a rectangle around the people
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 255, 0), 2)

                        # add the bounding box coordinates to the rectangles list
                        rects.append((startX, startY, endX, endY))

                # draw a horizontal line in the center of the frame -- once an
                # object crosses this line we will determine whether they were
                # moving 'up' or 'down'
                cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

                # use the centroid tracker to associate the (1) old object
                # centroids with (2) the newly computed object centroids
                objects = ct.update(rects)

                # loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    # check to see if a trackable object exists for the current
                    # object ID
                    to = trackableObjects.get(objectID, None)

                    # if there is no existing trackable object, create one
                    if to is None:
                        to = TrackableObject(objectID, centroid)

                    # otherwise, there is a trackable object so we can utilize it
                    # to determine direction
                    else:
                        # the difference between the y-coordinate of the *current*
                        # centroid and the mean of *previous* centroids will tell
                        # us in which direction the object is moving (negative for
                        # 'up' and positive for 'down')
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)

                        # check to see if the object has been counted or not
                        if not to.counted:
                            # if the direction is negative (indicating the object
                            # is moving up) AND the centroid is above the center
                            # line, count the object
                            if direction < 0 and centroid[1] < H // 2:
                                totalUp += 1
                                to.counted = True

                            # if the direction is positive (indicating the object
                            # is moving down) AND the centroid is below the
                            # center line, count the object
                            elif direction > 0 and centroid[1] > H // 2:
                                totalDown += 1
                                to.counted = True

                                ###change by wangao
                                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                             time.localtime(time.time()))
                                print('[EVENT] %s, 院子, 有人闯入禁止区域!!!' % current_time)


                    # store the trackable object in our dictionary
                    trackableObjects[objectID] = to

                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4,
                               (0, 255, 0), -1)

                # construct a tuple of information we will be displaying on the
                # frame
                info = [
                    # ("Up", totalUp),
                    ("Down", totalDown),
                    ("Status", status),
                ]

                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                image = cv2.resize(frame, (640, 480))   #压缩
                p.stdin.write(image.tostring())

                # show the output frame
                cv2.imshow("Frame", frame)

                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

                totalFrames += 1
                # fps.update()



def run():
    # 使用两个线程处理
    thread1 = Thread(target=Video, )
    thread1.start()
    time.sleep(2)
    thread2 = Thread(target=push_frame, )
    thread2.start()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()





