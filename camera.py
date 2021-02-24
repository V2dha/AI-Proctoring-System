import cv2
# from imutils.video import WebcamVideoStream
from yolov3 import *

class VideoCamera(object):
    def __init__(self):
        # self.stream = WebcamVideoStream(src=0).start()
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # self.stream.stop()
        self.video.release()

    def get_frame(self):
        # image = self.stream.read()
        _, image = self.video.read()
        yolo = YoloV3()
        load_darknet_weights(yolo, 'yolov3.weights')
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 320))
        img = img.astype(np.float32)
        img = np.expand_dims(img, 0)
        img = img / 255
        class_names = [c.strip() for c in open("classes.txt").readlines()]
        boxes, scores, classes, nums = yolo(img)
        count=0
        for i in range(nums[0]):
            if int(classes[0][i] == 0):
                count +=1
            if int(classes[0][i] == 67):
                print("Mobile Phone Detected")
            if int(classes[0][i] == 63):
                print("Laptop Detected")
            if int(classes[0][i] == 62):
                print("tvmonitor Detected")
            if int(classes[0][i] == 73):
                print("Book Detected")
        if count == 0:
            print('No person detected')
        elif count > 1: 
            print('More than one person detected')
        draw_outputs(image, (boxes, scores, classes, nums), class_names)
        ret, jpeg = cv2.imencode('.jpg', image)
        data = []
        data.append(jpeg.tobytes())
        return data   