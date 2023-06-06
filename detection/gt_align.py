
import os
import csv
from yolov5.modified_detect_simple import Yolo_Exec
import cv2
import time
import pybboxes as pbx
import pickle


def find_area(a):

    width = a[2] - a[0]
    height = a[3] - a[1]
    return width * height


def find_area_overlap(a,b):
    overlap = 0.0
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        overlap = dx*dy
    return overlap

# Get all gt files relating to object visibility
def get_obj_visibility_files(gt_dir):

    # List of ground truth files for object detections
    obj_detect_gt_filepaths = []

    # Now list all files and their ground truth
    for take_folder in os.listdir(gt_dir):
        current_take = int(take_folder.split("_")[1])
        current_dir = os.path.join(gt_dir, take_folder)
        # Iterate through each file in that directory (video and gt)
        for filename in os.listdir(current_dir):
            if "objectvisibility" in filename:
                # Now attach it to the current dir and save it
                obj_detect_gt_filepaths.append((current_take, current_dir, filename))

    # Sort gt filenames by ascending take
    obj_detect_gt_filepaths.sort(key= lambda x : x[0])

    return obj_detect_gt_filepaths



# Get first line of object detection
def get_first_X_gt_detections(gt_file, num_gt_candidates=20):

    gt_data_results = []
    previous_frame_index = 0
    # Read in the file's first line
    with open(gt_file, "r") as rfile:
        gt_reader = csv.reader(rfile, delimiter=',')

        # Iterate through every row of the CSV
        for row_index, row in enumerate(gt_reader):

            gt_data = []
            # If there are more than 10 items in the first row, it means it got stuck together
            if row_index == 0 and len(row) > 10:
                gt_data = row[9:]
            elif row_index > 0:
                gt_data = row

            if gt_data:
                # Parse gt data
                gt_data = parse_gt_data(gt_data)
                # Check if anything is negative.  If so, keep going.

                # If we didn't skip enough frames, then keep going
                # if gt_data["frame_count"] - previous_frame_index < separation:
                #     continue
                # else:
                #     previous_frame_index = gt_data["frame_count"]    

                # Do not use negative values in bounding boxes
                if any([x<0 for x in gt_data["xxyy"]]):
                    continue
                if gt_data["xxyy"][0] < 50 or gt_data["xxyy"][2] < 50: #  Make sure we apply the buffer
                    continue

                gt_data_results.append(gt_data)
        
            # If we get enough results, we return
            if num_gt_candidates==-1:
                continue
            if len(gt_data_results) >= num_gt_candidates:
                break

    return gt_data_results



# Parse gt data
def parse_gt_data(gt_data):

    # Remember, cam numbering starts from 1.
    if "NAI" in gt_data[3]:
        cam_num = int(gt_data[3].split("NAI")[1][:-1])
    else:
        cam_num = -1

    min_xxyy = eval(','.join(gt_data[9:11]))
    max_xxyy = eval(','.join(gt_data[11:]))
    xxyy = [max_xxyy[0], max_xxyy[1], min_xxyy[0], min_xxyy[1]]
    xxyy = [int(x) for x in xxyy]


    # In this CSV, there are only a few fields of interest:
    #  element 1, 2, 3, 6, and 9,10,11,12
    #  These refer to: scenario_time in seconds (1), frame count (2), camera name (3), object label (6)
    #   finally, the min xx,yy and max xx,yy coordinates (top left and bottom right)
    gt_data = {
        "scene_time": float(gt_data[1]),
        "frame_count": int(gt_data[2]),
        "cam_name": cam_num,
        "obj_label": int(gt_data[6]),
        "xxyy": xxyy,
    }   

    return gt_data





# Compare the results from yolo with the 
def compare_yolo_results(res_lines, image, pixel_width, pixel_height, coordinate_list, gt_data):

    issue_stop = False
    matched_data = None
    # Iterate through every YOLO result, and add them to detection_bboxes
    for line in res_lines:
    
        # Draw the detection boxes
        coordinates_line = line.split()
        box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),\
            float(coordinates_line[4])), from_type="yolo", to_type="voc", \
                                    image_size=(pixel_width, pixel_height))

        cv2.rectangle(image, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)
        # Get the detected obj class label
        obj_class_label = int(coordinates_line[0])
        # Draw the object class labels
        fontScale = 0.5
        color = (255, 153, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        image = cv2.putText(image, str(obj_class_label), (box_voc[2], box_voc[3]), font, 
                        fontScale, color, thickness, cv2.LINE_AA)


        # Now, we have to iterate through all our coordinates and see if it matches the gt
        for gt_index, gt_coords in enumerate(coordinate_list):

            # Get the gt obj label
            gt_label = gt_data[gt_index]["obj_label"]

            # Measure the overlap
            area_overlap = find_area_overlap(box_voc, gt_coords)
            measured_overlap = area_overlap / find_area(gt_coords)
            
            # See if we have a match with ground truth
            if obj_class_label == gt_label:
                if measured_overlap > 0.60:
                    matched_data = gt_data[gt_index]
                    issue_stop = True
                    break


    return issue_stop, matched_data

# Open a video and run the yolo detection on it
def run_yolo_detection(video_file, gt_data, yolo_model):

    offset_by_frame_count = 0
    
    # Now open up the video and get each frame
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Fix the coordinates
    gt_coordinates_list = []
    for gt_item in gt_data:
        gt_coordinates = gt_item["xxyy"]
        gt_coordinates[1] = pixel_height - gt_coordinates[1]
        gt_coordinates[3] = pixel_height - gt_coordinates[3]
        gt_coordinates_list.append(gt_coordinates)


    # Loop through each frame until we get a match
    gt_match = False
    frame_index = 0
    while not gt_match:
        ret, image = cap.read()


        # print(frame_index)
        # Now run yolo, and get the bounding boxes
        res_lines = yolo_model.run(image) #Run yolo

        # Draw all the ground truth bounding boxes
        for gt_coordinates in gt_coordinates_list:
            # Draw the gt bounding box
            cv2.rectangle(image, (gt_coordinates[0], gt_coordinates[1]), \
                (gt_coordinates[2], gt_coordinates[3]), (0, 0, 255), 1)


        issue_stop, matched_data = compare_yolo_results(res_lines, image, pixel_width, pixel_height, \
            gt_coordinates_list, gt_data)

        # cv2.imshow('image',image)
        # cv2.waitKey(1)

        if issue_stop:
            gt_match = True

            # Check the matched_data with the current frame index
            print(matched_data)
            print(frame_index)

            offset_by_frame_count = matched_data["frame_count"] - frame_index
            break


        # Add to our frame index
        frame_index += 1
    
    return offset_by_frame_count



def get_offsets_and_save_to_file(gt_files, save_file_name):

    # set up the YOLO model
    yolo_model = Yolo_Exec(weights= "../soartechDetectorV2.pt" \
        , imgsz=[1920],conf_thres=0.5, device="0", save_conf=True)


    frame_offset_dict = {}
    # Now, we must run detections on every take
    for tup in gt_files:

        video_take = tup[0]
        parent_folder = tup[1]
        gt_filename = tup[2]
        print(video_take)
        # Get the ground truth file and see what the detection should be
        gt_data = get_first_X_gt_detections(os.path.join(parent_folder, gt_filename))


        # if there's actually data, we keep going
        if gt_data:

            # Given the ground truth data, now we decide what video from this folder to check
            video_files = os.listdir(parent_folder)
            vfile_of_interest = ""
            for x in video_files:
                if "cam"+str(gt_data[0]["cam_name"]) in x:
                    vfile_of_interest = x
            
            # Run yolo detection on that file
            vfile_path = os.path.join(parent_folder, vfile_of_interest)
            frame_offset = run_yolo_detection(vfile_path, gt_data, yolo_model)

            # Save these frame offsets
            frame_offset_dict["take_"+str(video_take)] = frame_offset

    # Save the frame offset dict to a pickle file
    with open(save_file_name + ".pkl", "wb") as f:
        pickle.dump(frame_offset_dict, f)


#  This is the function for initializing all the ground truth bboxes for each frame and camera
#   This function returns a dict of:
#     {
#         camera_id: {
#                       corrected_frame_index : [list of (xy,xy,xy,xy,object class)]
#                    }     
#     }
#  The parameter 'fine_tune' is just adding a few frames to the offset to fix
#    some matching issues
def initialize_gt_bboxes(gt_items, offset, img_width, img_height, fine_tune=3):

    vertical_buffer = 10
    horizontal_buffer = 5
    # Get the coordinates, and make sure to add bounds
    gt_coordinates_list = []
    for gt_item in gt_items:
        gt_coordinates = gt_item["xxyy"]
        gt_coordinates[0] -= vertical_buffer
        gt_coordinates[1] = (img_height - gt_coordinates[1]) - horizontal_buffer
        gt_coordinates[2] += vertical_buffer
        gt_coordinates[3] = (img_height - gt_coordinates[3]) + horizontal_buffer
        gt_coordinates.append(gt_item["obj_label"])
        gt_coordinates_list.append(gt_coordinates)


    # Get a dict for every camera
    #  Each will have its own dictionary of frame counts
    cam_dict = {"cam1":{}, "cam2":{}, "cam3":{}}
    for g_i, gt_item in enumerate(gt_items):

        # Get the current camera
        current_camera = "cam" + str(gt_item["cam_name"])
        if current_camera not in cam_dict.keys():
            continue

        frame_value = gt_item["frame_count"] - offset + fine_tune

        if frame_value in cam_dict[current_camera]:
            cam_dict[current_camera][frame_value].append(gt_coordinates_list[g_i])
        else:
            cam_dict[current_camera][frame_value] = [gt_coordinates_list[g_i]]

    return cam_dict



# Open a video file and apply the ground truth
def apply_gt_and_load_video(vfilepath, gt_items, offset):

    # Start reading the video
    cap = cv2.VideoCapture(vfilepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    #  Get the dictionary for all camera data
    cam_dict = initialize_gt_bboxes(gt_items, offset, pixel_width, pixel_height)

    # Loop through each frame  and put up the ground truth bboxes based on the current frame
    current_frame_index = 0
    max_frame_index = 10000
    while current_frame_index < total_frames:
        ret, image = cap.read()

        coordinates = []
        if current_frame_index in cam_dict["cam1"]:
            coordinates = cam_dict["cam1"][current_frame_index]  # Currently only test with cam1

        # Draw all the ground truth bounding boxes
        for gt_coordinates in coordinates:
            # Draw the gt bounding box
            cv2.rectangle(image, (gt_coordinates[0], gt_coordinates[1]), \
                (gt_coordinates[2], gt_coordinates[3]), (0, 0, 255), 1)
    
        cv2.imshow('image',image)
        cv2.waitKey(1)

        current_frame_index += 1






# Run main function
if __name__=='__main__':

    # Get video folder
    gt_dir = "/media/brianw/1511bdc1-b782-4302-9f3e-f6d90b91f857/home/brianw/SoartechData/videos"
    # Get object visibility files
    gt_files = get_obj_visibility_files(gt_dir)

    # Get all offsets for the object visibility files and save to file
    get_offsets_and_save_to_file(gt_files, "offsets")
    asdf

    # # Open the pickle file
    # with open("offsets.pkl", "rb") as f:
    #     offset_data = pickle.load(f)
    #     print(data)

    # Iterate through each ground truth file and videos
    for gt_file in gt_files:

        current_take = "take_" + str(gt_file[0])

        if gt_file[0] != 205:
            continue


        # 205 - fc is 18581 while matched index is 573, so offset of 18008
        # 229 - fc is 72569 while matched index is 556, so offset of 72013
        # 250 - fc is 450590 while matched index is 536, so 450054

        # Take camera 1 as an example

        files_for_take = os.listdir(gt_file[1])
        video_file = [x for x in files_for_take if "cam1" in x]
        video_file = video_file[0]
        vfilepath = os.path.join(gt_file[1], video_file)

        gt_filepath =  os.path.join(gt_file[1], gt_file[2])
        gt_items = get_first_X_gt_detections(gt_filepath, num_gt_candidates=-1)  # -1 means get all data

        # apply_gt_and_load_video(vfilepath, gt_items, offset_data[current_take])
        apply_gt_and_load_video(vfilepath, gt_items, 18008)

        
    





