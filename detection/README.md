**load.py** is used to run the YOLO  model with inference. To train the model just use the original yolov5 files provided. Note that the yolov5/ directory in here is a modified version of the original repository

**Options**
 
* --yolo-weights: specify a *.pt file
* --video-file: specify a video file to run yolo over *.mp4
* --display: display yolo detections
* --conf-thres: specify a confidence threshold of yolo model
* --start: frame to start detection in video
* --end: frame to end detection in video
* --create-video: specify a *.mp4 video file name in order to create a video showing bounding boxes and tracks
* --yolo-synth-output: specify a directory where you have a file with yolo results to use them instead of activating yolo
* --save-raw-yolo-dir: specify a directory where to create files with the raw yolo results
* --save-tracking-dir: save results of only the bounding boxes being tracked
* --track_alg: choose which tracking algorithm to use. Right now default is ByteTracker
* --device: choose the gpu number
* --camera_id: number specifying camera id, can be whatever you want


**Example**

python load.py --yolo-weights yolov5/soartechDetectorV2.pt --video-file neuroplex_cam2_take_391_smoke.mp4 --display --device 1 --camera_id 1


**Other Files**

* yolov5/create_dataset.py: Extracts images from videos with smoke as well as the ground truth annotations from .csv file to create a dataset of images and text files per frame
* correct_labels.py: Correct the video and bounding box files with an offset showing the correct correspondance to frames
* yolov5/distribute.py: Based on this dataset just created, distribute files (create symlinks) based on a training-validation split
* run_load_smoke.py: Run the soartech yolo model on the videos with smoke and save the bounding boxes results
* combine.py: Combine the results of running the soartech yolo model with the ground truth files obtained using create_dataset.py
* get_frames.py: Dump frames from all videos with smoke
* extract_crops_multi.py: Based on images and bounding box annotations directories, crop the images using the bounding box information and save them in a directory
* train_classif.py: Using the extracted cropped images, trained an image classifier using HuggingFace and save the weights
* test_classif.py: Test the image classifier with a single image, directory of images, and get the accuracy metrics or the inference results.
* extract_crops.py: Run yolo model to get bounding boxes, use image classifier to classify bounding box crops, track the bounding boxes, and save the results in files
* use_ae_training.py: Run extract_crops.py on all videos with smoke, but specify the time periods at which to start and end detection based on atomic event information
* track_selection_ae.py: With the results of the extract_crops.py, try to figure out the correct bounding boxes using the track information and the atomic events information
* trackers/: all the files in this directory have to do with the tracking algorithms
