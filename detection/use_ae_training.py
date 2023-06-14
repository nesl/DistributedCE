import os
import re
import pdb

recce_vehicles = 4

#Get bounding boxes by running pretrained original yolo model

def get_range(ae_array, target_object_type, max_range):
	cameras_info = {}

	for k_ae in ae_array.keys():
		for ae_event in ae_array[k_ae]:
			if k_ae < recce_vehicles:
				object_type = 1
			else:
				object_type = 0
				
			if target_object_type > -1 and object_type != target_object_type:
				continue
		
			if ae_event[0] not in cameras_info:
				cameras_info[ae_event[0]] = [[ae_event[1], ae_event[2]]]
			else:
				for aea_idx,ae_prev_event in enumerate(cameras_info[ae_event[0]]):
					if (((not ae_event[2] < ae_prev_event[0]) or ae_event[2] == -1) and ((not ae_event[1] > ae_prev_event[1]) or ae_prev_event[1] == -1)):
					
						if max_range:
							if ae_event[1] < ae_prev_event[0]:
								cameras_info[ae_event[0]][aea_idx][0] = ae_event[1]
							if ae_event[2] > ae_prev_event[1] or ae_event[2] == -1:
								cameras_info[ae_event[0]][aea_idx][1] = ae_event[2]
						else:
							if ae_event[1] > ae_prev_event[0]:
								cameras_info[ae_event[0]][aea_idx][0] = ae_event[1]
							if ae_event[2] < ae_prev_event[1] and not ae_event[2] == -1:
								cameras_info[ae_event[0]][aea_idx][1] = ae_event[2]
	return cameras_info


def process_ae_file(ae_file):
	ae_f = open(ae_file)
	ae_lines = ae_f.readlines()
	
	ae_array = {}
	prev_camera = {}
	for ae_line in ae_lines:
		ae_temp = eval(ae_line.strip().split(":::")[1])
		
		for ae in ae_temp:
			for res in ae['results']:			
				if res[2] not in ae_array:
					ae_array[res[2]] = []
				if res[2] not in prev_camera:
					prev_camera[res[2]] = -1
					
				if prev_camera[res[2]] != ae['camera_id']:
					if ae_array[res[2]]:
						ae_array[res[2]][-1][2] = ae['time']-1
					ae_array[res[2]].append([ae['camera_id'],ae['time'], -1])
					
				prev_camera[res[2]] = ae['camera_id']
	return ae_array
	
		
							
	
				
domain_shift = "smoke"	

root_path = "Elements/results/using_gt/CE1_" + domain_shift + "_50_True_True_True/"

video_path = "Elements/"
write_path = "tracks_weakly_supervised/"

allowed_cameras = [2,3]

cam_num_file = re.compile(r'_cam(\d+)_')

for path_dir in os.listdir(root_path):

	
	ae_array = process_ae_file(root_path + path_dir + "/ce_output.txt")

	print(ae_array)

	resulting_range = get_range(ae_array, -1, True)

	print(resulting_range)
	
	
	take_dir = video_path+"take_" + path_dir + "/"
	for vid in os.listdir(take_dir):
	
		vid_name = take_dir+vid
		
		if '.mp4' in vid and domain_shift in vid:
			mo = cam_num_file.search(vid)
			
			camera = mo.group(1)
			
			if int(camera) not in allowed_cameras:
				continue
			
			
			for r_idx,ranges in enumerate(resulting_range[str(int(camera)-1)]):
			
				write_dir = "take_" + path_dir + "_" + camera + "_" + str(r_idx)
			
				command = "python extract_crops.py --vid-name " + vid_name + " --start " + str(ranges[0]) + " --write-gt-dir " + write_path + write_dir + " --conf-thres 0.01 --create-video " + write_path + write_dir + ".mp4 --yolo-weights ~/Downloads/yolov5x6.pt --model img_classifier/checkpoint-66"
			
				if ranges[1] > -1:
					command += " --end " + str(ranges[1])
				

				os.system(command)
			


