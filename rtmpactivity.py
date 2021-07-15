# -*- coding: utf-8 -*-

"""
义工交互检测

用法:


"""

import argparse
import time

import cv2
import queue
from threading import Thread
import datetime, _thread
import imutils
import subprocess

import tensorflow as tf
import keras
from flask import Flask, render_template, Response, request
from imutils.video import WebcamVideoStream
from keras.models import load_model
from A_Final_Sys.oldcare.facial import FaceUtil
from A_Final_Sys.oldcare.camera import VideoCamera
from A_Final_Sys.oldcare.utils import Time_Controller
from A_Final_Sys.oldcare.utils import Fence_Tools
from A_Final_Sys.oldcare.utils import getIP
from flask_cors import *
from A_Final_Sys.oldcare.utils import fileassistant
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import keras
from flask import Flask, render_template, Response, request
from imutils.video import WebcamVideoStream
from keras.models import load_model
import subprocess as sp
from scipy.spatial import distance as dist
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import imutils
import numpy as np
import argparse


VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ANGLE = 20
# 全局常量

limit_time = 2
# 使用线程锁，防止线程死锁
mutex = _thread.allocate_lock()
# 存图片的队列
frame_queue = queue.Queue()

# rtmpUrl = "rtmp://localhost:1935/rtmplive"
rtmpUrl = "rtmp://192.168.140.109:1935/rtmplive"

command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', "{}x{}".format(640, 480),  # 图片分辨率
           '-r', str(15.0),  # 视频帧率
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
    #全局变量
    pixel_per_metric = None
    model_path = 'models/face_recognition_hog.pickle'
    people_info_path = 'info/people_info.csv'

    # 全局常量
    FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480
    ACTUAL_DISTANCE_LIMIT = 100  # cm

    id_card_to_name, id_card_to_type = fileassistant.get_people_info(people_info_path)

    # 加载模型
    faceutil = FaceUtil(model_path)

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
        if frame_queue.empty() != True:
            counter = 0

            while True:
                counter += 1
                image = frame_queue.get()
                if counter <= 10:  # 放弃前10帧
                    continue
                image = cv2.flip(image, 1)  #镜像翻转

                frame = imutils.resize(image,
                                       width=VIDEO_WIDTH,
                                       height=VIDEO_HEIGHT)  # 压缩，为了加快识别速度

                face_location_list, names = faceutil.get_face_location_and_name(frame)

                people_type_list = list(set([id_card_to_type[i] for i in names]))

                volunteer_centroids = []
                old_people_centroids = []
                old_people_name = []

                # loop over the face bounding boxes
                for ((left, top, right, bottom), name) in zip(face_location_list, names):  # 处理单个人

                    person_type = id_card_to_type[name]
                    # 将人脸框出来
                    rectangle_color = (0, 0, 255)
                    if person_type == 'old_people':
                        rectangle_color = (0, 0, 128)
                    elif person_type == 'employee':
                        rectangle_color = (255, 0, 0)
                    elif person_type == 'volunteer':
                        rectangle_color = (0, 255, 0)
                    else:
                        pass
                    cv2.rectangle(frame, (left, top), (right, bottom),
                                  rectangle_color, 2)

                    if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
                        continue

                    if person_type == 'volunteer':  # 如果检测到有义工存在
                        # 获得义工位置
                        volunteer_face_center = (int((right + left) / 2),
                                                 int((top + bottom) / 2))
                        volunteer_centroids.append(volunteer_face_center)

                        cv2.circle(frame,
                                   (volunteer_face_center[0], volunteer_face_center[1]),
                                   8, (255, 0, 0), -1)

                    elif person_type == 'old_people':  # 如果没有发现义工
                        old_people_face_center = (int((right + left) / 2),
                                                  int((top + bottom) / 2))
                        old_people_centroids.append(old_people_face_center)
                        old_people_name.append(name)

                        cv2.circle(frame,
                                   (old_people_face_center[0], old_people_face_center[1]),
                                   4, (0, 255, 0), -1)
                    else:
                        pass

                    # 人脸识别和表情识别都结束后，把表情和人名写上 (同时处理中文显示问题)
                    img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_PIL)
                    final_label = id_card_to_name[name]
                    draw.text((left, top - 30), final_label,
                              font=ImageFont.truetype("C:\\WINDOWS\\Fonts\\simhei.ttf", 40),
                              fill=(255, 0, 0))  # linux
                    # 转换回OpenCV格式
                    frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

                # 在义工和老人之间划线
                for i in volunteer_centroids:
                    for j_index, j in enumerate(old_people_centroids):
                        pixel_distance = dist.euclidean(i, j)
                        face_pixel_width = sum([i[2] - i[0] for i in face_location_list]) / len(face_location_list)
                        pixel_per_metric = face_pixel_width / FACE_ACTUAL_WIDTH
                        actual_distance = pixel_distance / pixel_per_metric

                        if actual_distance < ACTUAL_DISTANCE_LIMIT:
                            cv2.line(frame, (int(i[0]), int(i[1])),
                                     (int(j[0]), int(j[1])), (255, 0, 255), 2)
                            label = 'distance: %dcm' % (actual_distance)
                            cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 2)

                            current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                         time.localtime(time.time()))
                            print('[EVENT] %s, 房间桌子, %s 正在与义工交互.' % (
                            current_time, id_card_to_name[old_people_name[j_index]]))


                # show our detected faces along with smiling/not smiling labels
                cv2.imshow("Checking Volunteer's Activities", frame)

                # Press 'ESC' for exiting video
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

                image = cv2.resize(frame, (640, 480))  # 压缩
                p.stdin.write(image.tostring())




def run():
    # 使用两个线程处理
    thread1 = Thread(target=Video, )
    thread1.start()
    time.sleep(4)
    thread2 = Thread(target=push_frame, )
    thread2.start()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()





