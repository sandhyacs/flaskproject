import sys
import time
import json
from flask_cors import CORS
from flask import Flask, request
import base64
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
from utils import label_map_util
from utils import visualization_utils_colorimage as vis_util
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

PATH_TO_CKPT = "./model/frozen_inference_graph.pb"
PATH_TO_LABELS = './protos/ear_label.pbtxt'
landmark_model = "./model/my_model_50e_8b.h5"

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowEarDector(object):
    def __init__(self, PATH_TO_CKPT):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True

    def run(self, image):
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time

        return (boxes, scores, classes, num_detections)

def imgBase64decode(data):
    nparr = np.fromstring(base64.b64decode(data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def load_landmakr_model(model):
    model = load_model(model)
    return model
landmark_model = load_landmakr_model(landmark_model)

app = Flask(__name__)
CORS(app)

@app.route('/EarLandmarkDetection', methods=["POST", "GET"])

def earTracking():
    try:
        coordinates = {}
        bounding_box = {}
        base64Data = request.form["image"]
        img = imgBase64decode(base64Data)
        tDetector = TensoflowEarDector(PATH_TO_CKPT)
        [h, w] = img.shape[:2]
        (boxes, scores, classes, num_detections) = tDetector.run(img)
        box=vis_util.bbox_coordinates(img, np.squeeze(boxes), np.squeeze(scores), use_normalized_coordinates=True)
        ymin, xmin, ymax, xmax = box
        (left, right, top, bottom) = (xmin * w, xmax * w, ymin * h, ymax * h)
        bounding_box["left"] = xmin * w
        bounding_box["right"] = xmax * w
        bounding_box["top"] = ymin * h
        bounding_box["bottom"] = ymax * h
        left, right, top, bottom = int(left), int(right), int(top), int(bottom)
        roi = img[top:bottom, left:right]
        [roi_h, roi_w]=roi.shape[:2]
        roi_h = roi_h+30
        img1 = cv2.resize(roi, (224, 224))
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        prediction = landmark_model.predict(x)
        coordinates["x"] = round(((prediction[0][0]*roi_w)+left),2)
        coordinates["y"] = round(((prediction[0][1]*roi_h)+top), 2)
        return json.dumps({"Status": "SUCCESS", "Bounding Box":bounding_box, "Coordinates" : coordinates})
    except Exception as e:
        return json.dumps({"status": "Not Detected"})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
