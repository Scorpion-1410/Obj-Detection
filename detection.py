from imageai.Detection import ObjectDetection
from IPython.display import Image
import os
import sys

execution_path = os.getcwd()
argument_path = sys.argv[1]
argument_name = sys.argv[2]

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(argument_path , argument_name), output_image_path=os.path.join(argument_path , "new_"+argument_name))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

Image("imgnew.jpeg")
