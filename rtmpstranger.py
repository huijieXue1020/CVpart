# -*- coding: utf-8 -*-

"""
测试rtmp推流

用法:


"""

import argparse
import os
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
import subprocess as sp
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    facial_recognition_model_path = 'models/face_recognition_hog.pickle'
    facial_expression_model_path = 'models/miniGOOGLE_Adam_100_7.hdf5'

    people_info_path = 'info/people_info.csv'
    facial_expression_info_path = 'info/facial_expression_info.csv'

    output_stranger_path = 'supervision/strangers'
    output_smile_path = 'supervision/smile'

    facial_expression_model = load_model(facial_expression_model_path)
    faceutil = FaceUtil(facial_recognition_model_path)

    python_path = 'F:\\Program Files\\anaconda\\envs\\tt\\python'

    # 得到 ID->姓名的map 、 ID->职位类型的map、
    # 摄像头ID->摄像头名字的map、表情ID->表情名字的map
    id_card_to_name, id_card_to_type = fileassistant.get_people_info(
        people_info_path)
    facial_expression_id_to_name = fileassistant.get_facial_expression_info(
        facial_expression_info_path)



    WIDTH = 640
    HEIGHT = 480
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
        stranger_time_controller = Time_Controller()
        face_time_controller = Time_Controller()

        if frame_queue.empty() != True:
            counter = 0

            while True:
                counter += 1
                image = frame_queue.get()
                if counter <= 10:  # 放弃前10帧
                    continue
                image = cv2.flip(image, 1)  #镜像翻转

                frame = imutils.resize(image, width=WIDTH,
                                       height=HEIGHT)  # 压缩，加快识别速度
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

                face_location_list, names = faceutil.get_face_location_and_name(frame)

                # 得到画面的四分之一位置和四分之三位置，并垂直划线
                one_fourth_image_center = (int(WIDTH / 4),
                                           int(HEIGHT / 4))
                three_fourth_image_center = (int(WIDTH / 4 * 3),
                                             int(HEIGHT / 4 * 3))

                cv2.line(frame, (one_fourth_image_center[0], 0),
                         (one_fourth_image_center[0], HEIGHT),
                         (0, 255, 255), 1)
                cv2.line(frame, (three_fourth_image_center[0], 0),
                         (three_fourth_image_center[0], HEIGHT),
                         (0, 255, 255), 1)

                # 处理每一张识别到的人脸
                for ((left, top, right, bottom), name) in zip(face_location_list,
                                                              names):

                    # 将人脸框出来
                    rectangle_color = (0, 0, 255)
                    if id_card_to_type[name] == 'old_people':
                        rectangle_color = (0, 0, 128)
                    elif id_card_to_type[name] == 'employee':
                        rectangle_color = (255, 0, 0)
                    elif id_card_to_type[name] == 'volunteer':
                        rectangle_color = (0, 255, 0)
                    else:
                        pass
                    cv2.rectangle(frame, (left, top), (right, bottom),
                                  rectangle_color, 2)

                    # 陌生人检测逻辑
                    if 'Unknown' in names:  # alert
                        if stranger_time_controller.strangers_timing == 0:  # just start timing
                            stranger_time_controller.set_stranger_timing(1)
                            stranger_time_controller.start_stranger_time()
                        else:  # already started timing
                            strangers_end_time = time.time()
                            difference = strangers_end_time - stranger_time_controller.strangers_start_time

                            current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                            if difference < stranger_time_controller.strangers_limit_time:
                                print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                            else:  # strangers appear
                                event_desc = '陌生人出现!!!'
                                event_location = '房间'
                                print('[EVENT] %s, 房间, 陌生人出现!!!' % current_time)
                                cv2.imwrite(os.path.join(output_stranger_path,
                                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                            frame)  # snapshot

                                #########
                                pic_path = os.path.join(output_stranger_path,
                                                        'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                                # insert into database
                                command1 = '%s inserting.py --event_desc %s --event_type 2 --event_location %s --pic_path %s' % (
                                    python_path, event_desc, event_location, pic_path)
                                p1 = subprocess.Popen(command1, shell=True)

                                # 开始陌生人追踪
                                unknown_face_center = (int((right + left) / 2),
                                                       int((top + bottom) / 2))

                                cv2.circle(frame, (unknown_face_center[0],
                                                   unknown_face_center[1]), 4, (0, 255, 0), -1)

                                direction = ''
                                # face locates too left, servo need to turn right,
                                # so that face turn right as well
                                if unknown_face_center[0] < one_fourth_image_center[0]:
                                    direction = 'right'
                                elif unknown_face_center[0] > three_fourth_image_center[0]:
                                    direction = 'left'

                                # adjust to servo
                                if direction:
                                    print('摄像头需要 turn %s %d 度' % (direction, 20))

                    else:  # everything is ok
                        stranger_time_controller.set_stranger_timing(0)

                    # 表情检测逻辑
                    # 如果不是陌生人，且对象是老人
                    if name != 'Unknown' and id_card_to_type[name] == 'old_people':
                        # 表情检测逻辑
                        roi = gray[top:bottom, left:right]
                        roi = cv2.resize(roi, (28, 28))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        # determine facial expression
                        arr = list(facial_expression_model.predict(roi)[0])
                        labels = ['anger', 'disgust', 'fear', 'happy', 'normal', 'sad', 'surprised']
                        max_prediction = max(arr)
                        index = arr.index(max_prediction)
                        if index == -1:
                            facial_expression_label = labels[4]
                        else:
                            facial_expression_label = labels[index]

                        if facial_expression_label == 'happy':  # alert
                            if face_time_controller.facial_expression_timing == 0:  # just start timing
                                face_time_controller.set_facial_expression_timing(1)
                                face_time_controller.start_facial_expression__time()
                            else:  # already started timing
                                facial_expression_end_time = time.time()
                                difference = facial_expression_end_time - face_time_controller.facial_expression_start_time

                                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                             time.localtime(time.time()))
                                if difference < face_time_controller.facial_expression_limit_time:
                                    print('[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (
                                    current_time, id_card_to_name[name], difference))
                                else:  # he/she is really smiling
                                    event_desc = '%s正在笑' % (id_card_to_name[name])
                                    event_location = '房间'
                                    print('[EVENT] %s, 房间, %s正在笑.' % (current_time, id_card_to_name[name]))
                                    cv2.imwrite(os.path.join(output_smile_path,
                                                             'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                                frame)  # snapshot

                                    pic_path = os.path.join(output_stranger_path,
                                                            'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S')))
                                    # insert into database
                                    command2 = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                                        python_path, event_desc, event_location, pic_path)
                                    p2 = subprocess.Popen(command2, shell=True)

                        else:  # everything is ok
                            face_time_controller.set_facial_expression_timing(0)

                    else:  # 如果是陌生人，则不检测表情
                        facial_expression_label = ''

                    # 人脸识别和表情识别都结束后，把表情和人名写上
                    # (同时处理中文显示问题)
                    img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    draw = ImageDraw.Draw(img_PIL)
                    final_label = id_card_to_name[name] + ': ' + facial_expression_id_to_name[
                        facial_expression_label] if facial_expression_label else id_card_to_name[name]
                    draw.text((left, top - 30), final_label,
                              font=ImageFont.truetype("C:\Windows\Fonts\simsun.ttc", 30),
                              fill=(255, 0, 0))  # windows

                    # 转换回OpenCV格式
                    frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

                image = cv2.resize(frame, (640, 480))   #压缩
                p.stdin.write(image.tostring())

                # show our detected faces along with smiling/not smiling labels
                cv2.imshow("Face Recognition", frame)

                # Press 'ESC' for exiting video
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break







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





