import re
import os

#execute yolo model with smokey videos and save results

root_directory = "/media/nesl/Elements/data/"

cam_num_file = re.compile(r'_cam(\d+)_')


for dir_name in os.listdir(root_directory):
    take_dir = root_directory+dir_name+'/'
    if any("smoke" in take_dir+vid for vid in os.listdir(take_dir)):
        previous_ground_truth = []
        ground_truth_list = []
        cameras = []
        ground_truth_file = ""
        for vid in os.listdir(take_dir):
            vid_name = take_dir+vid
            if '.mp4' in vid and "smoke" in vid:
                
                mo = cam_num_file.search(vid)
                
                if int(mo.group(1)) > 1:
                    os.system("python load.py --yolo-weights yolov5_original/soartechDetectorV2.pt --video-file " + vid_name + " --camera_id 2 --device 0 --save-raw-yolo-dir results_smoke/" + dir_name[5:] + "_" + mo.group(1))

