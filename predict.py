import os
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import argparse
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger
import tensorflow as tf
from model import segunet
from generator import data_gen_small

class LIPSegmentationModel():
    def __init__(self, input_shape = (512, 512, 3)):
        self.model = segunet(input_shape, 20, 3, (2, 2), "softmax")
        self.model.load_weights("resources/checkpoints/checkpoint_e02.hdf5")
        self.input_shape = input_shape
        self.cascade = cv2.CascadeClassifier("resources/haarcascade_fullbody.xml")
        print(self.model.summary())

    def predict(self, input_img):
        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        rects = self.cascade.detectMultiScale(gray)
        img = input_img.copy()
        for (x,y,w,h) in rects:
            input_img = img[x:np.clip(x + w, 2, None), y:np.clip(y+h, 2, None)]
            print(input_img.shape)
            cv2.imshow("", input_img)
            cv2.waitKey()
            o_shape = input_img.shape
            resized_img = cv2.resize(input_img, self.input_shape[:2])
            array_img = img_to_array(resized_img) / 255
            prediction = self.model.predict(np.array([array_img]))
            prediction = np.reshape(prediction, (1,) + self.input_shape[:2] + (20,))[0][:,:,1:]
            prediction = np.argmax(prediction, axis=2).astype(np.uint8)
            return cv2.resize(prediction, o_shape[:2][::-1], interpolation=cv2.INTER_NEAREST)



if __name__ == '__main__':
    with tf.Graph().as_default():
        session = tf.Session("")
        model = LIPSegmentationModel()
        cap = cv2.VideoCapture("resources/images/test_movie.avi")
        i = 6000
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            print(i)
            i += 100
            ret, frame = cap.read()
            if frame is None:
                break
            mask = model.predict(frame)
            if mask is not None:
                result = np.zeros_like(frame)
                result[np.where(mask > 0)] = [200, 200 ,200]
                print(mask.shape, mask.dtype, np.mean(mask))
                cv2.imshow("input", frame)
                cv2.imshow("mask", mask * 20)
                cv2.waitKey()