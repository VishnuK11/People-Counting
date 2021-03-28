# People-Counting Crowds Using Yolov4+DeepSort

This project involves Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. The code is customized to track people in crowds specifically, however, can be modified for other object categories. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. Deep SORT is an object tracking module. We can take the output of YOLOv4 feed these object detections into Deep SORT in order to create a highly accurate object tracker.

## Demo of Object Tracker on Persons
<p align="center"><img src="data/helpers/demo.gif"\></p>

## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

## Downloading Official YOLOv4 Pre-trained Weights
Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

Copy and paste yolov4.weights from your downloads folder into the 'data' folder of this repository.

If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate, download file here: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

## Running the Tracker with YOLOv4
To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.

## Running Yolov4 + Deep Sort Object Tracker and People Counter

python object_tracker.py --weights ./checkpoints/yolov4-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/outputs.avi 

If running on collab, add ! prefix before python for the above line.
```

## Resulting Video
As mentioned above, the resulting video will save to wherever you set the ``--output`` command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the ``--output_format`` flag, by default it is set to AVI codec which is XVID.

Example video showing tracking of all coco dataset classes:
<p align="center"><img src="data/helpers/all_classes.gif"\></p>

## Filter Classes that are Tracked by Object Tracker
By default the code is setup to track only class - person. However, you can easily adjust a few lines of code in order to track more than 1 or combination of the 80 classes. To filter a custom selection of classes all you need to do is comment out line 159 and uncomment out line 162 of [object_tracker.py] The classes can be any of the 80 that the model is trained on, see which classes you can track in the file [data/classes/coco.names]

## Command Line Args Reference
  
 object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
```

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
