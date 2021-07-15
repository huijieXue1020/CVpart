# -*- coding: utf-8 -*-

'''
从视频中把图像分离出来
'''

import cv2
import os

input_video_path = 'videos/Home_02/Videos'
output_path = 'images'
prefix = 'Home_02_'

history = 20  # 训练帧数
frames = 0
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)  # 背景减除器，设置阴影检测
bs.setHistory(history)

for _, _, filelist in os.walk(input_video_path):
    pass


image_counter = 1
video_counter = 1
for i in filelist:
    filename = os.path.join(input_video_path, i)
    
    vs = cv2.VideoCapture(filename)

    
    while True:
        # grab the current frame
        (grabbed, frame) = vs.read()
        if filename and not grabbed:
            break

        fg_mask = bs.apply(frame)  # 获取 foreground mask
        if frames < history:
            frames += 1
            continue
        th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)  # 获取所有检测框
        cv2.imwrite(os.path.join(output_path, prefix + '%d.jpg' % image_counter), dilated)
        
        image_counter += 1
        
    vs.release()
    
    print('processed %d/%d videos' %(video_counter, len(filelist)))
    
    video_counter += 1


cv2.destroyAllWindows()