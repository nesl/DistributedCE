load.py is used to run the YOLO  model with inference. To train the model just use the original yolov5 files provided.

Options: 
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


Example:

python load.py --yolo-weights yolov5/soartechDetectorV2.pt --video-file neuroplex_cam2_take_391_smoke.mp4 --display --device 1 --camera_id 1
