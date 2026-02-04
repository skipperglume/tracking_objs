
### Using YOLO models to do car detection and use information for tracking
```bash
python3 -m venv car_follow
. car_follow/bin/activate
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install pyyaml
pip install jupyter
pip install ultralytics
pip install opencv-python
```


```bash

python -m pip install opencv-contrib-python
pip show opencv 


# pip uninstall opencv-python
# pip uninstall ultralytics
pip install opencv-contrib-python

pip list | g cv
pip install ultralytics
```

### NOTE: If code does not work with creation of trackers - `opencv-python` (this package does not have trackers) is used instead of `opencv-contrib-python` (this package does have trackers!). Small machinations with pip uninstall and install should do the trick:

```bash
pip uninstall -y \
  opencv-python \
  opencv-python-headless \
  opencv-contrib-python \
  opencv-contrib-python-headless


pip install opencv-contrib-python
```


### Display of results:

 - ![One](tracking_examples/tracked_output.gif)


### Seed estimation:
 - Is done with assumption camera is hold still




Classes that model can detect:
```python
{0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}
```