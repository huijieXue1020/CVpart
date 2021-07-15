# -*- coding: utf-8 -*-
"""
陌生人识别模型和表情识别模型的结合的主程序
"""

# 导入包
from PIL import Image, ImageDraw, ImageFont
from keras.preprocessing.image import img_to_array
import cv2
import time
import numpy as np
import os
import imutils
import subprocess


def check_stranger_and_emotion(frame, output_stranger_path, output_smile_path,
                               id_card_to_name, id_card_to_type,
                               facial_expression_id_to_name,
                               stranger_time_controller,
                               face_time_controller,
                               faceutil, facial_expression_model):
    WIDTH = 640
    HEIGHT = 480

    frame = imutils.resize(frame, width=WIDTH,
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
                                             'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), frame)  # snapshot

                    # insert into database
                    # command = '%s inserting.py --event_desc %s --event_type 2 --event_location %s' % (
                    #     python_path, event_desc, event_location)
                    # p = subprocess.Popen(command, shell=True)

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
                        print('[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (current_time, id_card_to_name[name], difference))
                    else:  # he/she is really smiling
                        event_desc = '%s正在笑' % (id_card_to_name[name])
                        event_location = '房间'
                        print('[EVENT] %s, 房间, %s正在笑.' % (current_time, id_card_to_name[name]))
                        cv2.imwrite(os.path.join(output_smile_path,
                                                 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                    frame)  # snapshot

                        # insert into database
                        # command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                        #     python_path, event_desc, event_location, int(name))
                        # p = subprocess.Popen(command, shell=True)

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
                  font=ImageFont.truetype("C:\\Windows\\Fonts\\SIMYOU.TTF", 30),
                  fill=(255, 0, 0))  # windows

        # 转换回OpenCV格式
        frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return frame
