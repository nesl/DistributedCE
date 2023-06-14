import os
import cv2
import pybboxes as pbx
import pdb
import numpy as np


#Select tracks based on context

def sort_file(e):
	return int(e[:-4])

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
					for ae_times_idx, ae_times in enumerate(ae_array[res[2]]):
						if ae_times[0] == prev_camera[res[2]] and ae_times[3] < 0:
							ae_array[res[2]][ae_times_idx][3] = ae['time']-1
							

						
				for w_idx,watchbox in enumerate(res[0]):
					ae_idx = -1
					for ae_times_idx, ae_times in enumerate(ae_array[res[2]]):
						if ae_times[0] == ae['camera_id'] and ae_times[1] == watchbox:
							ae_idx = ae_times_idx
							
					if ae_idx > -1:
						if res[1][w_idx]:
							ae_array[res[2]][ae_idx][2] = ae['time']
						else:
							ae_array[res[2]][ae_idx][3] = ae['time']
					else:
						if res[1][w_idx]:
							ae_array[res[2]].append([ae['camera_id'], watchbox, ae['time'], -1])
						else:
							ae_array[res[2]].append([ae['camera_id'], watchbox, 0, ae['time']])
							
				prev_camera[res[2]] = ae['camera_id']
	return ae_array
	
	
def FindPoints(watchbox, point):

	conditions = (point[:,0] > watchbox[0]) & (point[:,0] < watchbox[2]) & (point[:,1] > watchbox[1]) & (point[:,1] < watchbox[3])
	return conditions


def FindPoint(watchbox, point):

	conditions = (point[0] > watchbox[0]) and (point[0] < watchbox[2]) and (point[1] > watchbox[1]) and (point[1] < watchbox[3])
	return conditions

cameras = ['0','1','2']

watchbox = {}

for c in cameras:
	watchbox[c] = {}




watchbox['2'][0] = [200,1,1919,1079]
watchbox['2'][1] = [213,274,772,772]
watchbox['2'][2] = [816,366,1200,725]
watchbox['2'][3] = [1294,290,1881,765]
watchbox['1'][0] = [413,474,1072,772]
watchbox['0'][0] = [1,1,1919,1079]


tracks_path = "tracks_weakly_supervised/"

take = "351"
camera = "3"

take_folder = "take_" + take + "_" + camera + "_0"

domain_shift = "smoke"	
root_path = "Elements/results/using_gt/CE1_" + domain_shift + "_50_True_True_True/" + take + "/ce_output.txt"



cap = cv2.VideoCapture(tracks_path+take_folder+".mp4")

pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))




take_folder += '/'

frames_files = os.listdir(tracks_path+take_folder)

frames_files.sort(key=sort_file)

track_dict = {}

for ffile in frames_files:
	fe = open(tracks_path+take_folder+ffile)
	
	fe_lines = fe.readlines()
	
	for line in fe_lines:
		strip_line = line.strip().split(" ")
		
		track_id = int(strip_line[5])
		
		frame_num = int(ffile[:-4])
		
		
		
		if track_id not in track_dict:
			track_dict[track_id] = {"frame":np.array([]),"bbox":np.array([]), "class":[]}
			
		if frame_num in track_dict[track_id]["frame"]:
			continue
			
		
		        
		track_dict[track_id]["frame"] = np.append(track_dict[track_id]["frame"],frame_num)

		box_voc = pbx.convert_bbox((float(strip_line[1]),float(strip_line[2]),float(strip_line[3]),float(strip_line[4])), from_type="yolo", to_type="voc", image_size=(pixel_width,pixel_height))
		#track_dict[track_id]["bbox"] = np.append(track_dict[track_id]["bbox"], box_voc)
		if track_dict[track_id]["bbox"].size == 0:
			track_dict[track_id]["bbox"] = np.expand_dims(np.array(box_voc),axis=0)
		else:
			track_dict[track_id]["bbox"] = np.concatenate((track_dict[track_id]["bbox"] , np.expand_dims(np.array(box_voc),axis=0)),axis=0)
		track_dict[track_id]["class"].append(int(strip_line[0]))
			
	
camera = str(int(camera)-1)

tolerance = 5

aes = process_ae_file(root_path)

frame_aes = [{w:{"count":0,"tracks":[]} for w in watchbox[camera].keys()} for r in range(num_frames)]


print(aes)


ae_ranges = {}
ae_results = {}
min_frame = -1
for ce in aes.keys():

	events_array = []
	for events in aes[ce]:
		if events[0] == camera:
			events_array.append(events[1])
			
			if min_frame == -1 or min_frame > events[2]:
				min_frame = events[2]
			
	print(ce, events_array)
	
for ce in aes.keys():
	for events in aes[ce]:
		if events[0] == camera:
			if ce not in ae_ranges:
				ae_ranges[ce] = {}
				ae_results[ce] = {}
			
			end_range = num_frames+min_frame
			if events[3] > 0:
				end_range = events[3]
			
			ae_ranges[ce][events[1]]= np.array(list(range(events[2],end_range+1)))
			ae_results[ce][events[1]] = {}

for t_key in track_dict.keys():

	track_dict[t_key]["unique_class"] = np.bincount(track_dict[t_key]["class"]).argmax()
	for ce in ae_ranges.keys():
		for evt in ae_ranges[ce].keys():
			t_bool = np.in1d(track_dict[t_key]["frame"],ae_ranges[ce][evt])
			#t_indices = track_dict[track_id]["frame"][t_bool]
			#pdb.set_trace()


			ae_results[ce][evt][t_key] = sum(FindPoints(watchbox[camera][evt],track_dict[t_key]["bbox"][t_bool]))/len(ae_ranges[ce][evt])

				
			
#print(ae_results)

for ce in ae_results.keys():
	
	if int(ce) < 4:
		ae_class = 1
	else:
		ae_class = 0
			
	for evt in ae_results[ce].keys():
		max_score = 0
		max_key = -1

		
		key_val = np.array([[t_key,ae_results[ce][evt][t_key]] for t_key in ae_results[ce][evt].keys() if track_dict[t_key]["unique_class"] == ae_class])
		
		ind = np.argpartition(key_val[:,1], -5)[-5:]
		ind = np.flip(ind[np.argsort(key_val[ind,1])])
		
		
		
		"""
		for t_key in ae_results[ce][evt].keys():
		
			#detected_class = np.bincount(track_dict[t_key]["class"]).argmax()
			
			if detected_class == ae_class:
			
				res = ae_results[ce][evt][t_key]
				
				if res > max_score:
					max_score = res
					max_key = t_key
		"""
		print(key_val[ind,0], "matches with", ce, "watchbox", evt, "score", key_val[ind,1])
		

"""
	
for f_idx in range(len(frame_aes)):
	for ce in aes.keys():
		for events in aes[ce]:
			if events[0] == camera:
				real_frame_idx = f_idx+min_frame
				if real_frame_idx > events[2] and (real_frame_idx < events[3] or events[3] < 0):
					frame_aes[f_idx][events[1]]["count"] += 1
					
				for t_key in track_dict.keys():
					if real_frame_idx in track_dict[t_key]["frame"]:
						rf_idx = track_dict[t_key]["frame"].index(real_frame_idx)
						if FindPoint(watchbox[camera][events[1]],track_dict[t_key]["bbox"][rf_idx]):
							frame_aes[f_idx][events[1]]["tracks"].append(t_key)
		
							

print(frame_aes)
"""
"""

for t_key in track_dict.keys():
	for ce in aes.keys():
		
		#if (t_key == 3 or t_key == 13 or t_key == 14) and (ce == 2 or ce == 3):
		#	pdb.set_trace()
		
		check = []
		for events in aes[ce]:
			if events[0] == camera:
			
				
			
				try:
					b_array = np.where((np.array(track_dict[t_key]["frame"]) <= events[2]+tolerance) & (np.array(track_dict[t_key]["frame"]) >= events[2]-tolerance))[0]
				except:
					pdb.set_trace()
				
				if (t_key == 3 or t_key == 13 or t_key == 14) and (ce == 2 or ce == 3):
					print("Entrance", " Watchbox: ", events[1], " Detected Track: ", t_key, " Original Track: ", ce, np.array(track_dict[t_key]["frame"])[b_array], " Reference: ", events[2])
				
				if b_array.size > 0:
					b_idx = b_array[0]

					bboxes = np.array(track_dict[t_key]["bbox"])[b_array][:2]
					
					if len(bboxes.shape) == 1:
						bboxes = np.expand_dims(bboxes,0)
					
					check.append(any(FindPoints(watchbox[camera][events[1]],bboxes)))
				else:
					check.append(False)
				
				if events[3] > -1:

					b_array = np.where((np.array(track_dict[t_key]["frame"]) <= events[3]+tolerance) & (np.array(track_dict[t_key]["frame"]) >= events[3]-tolerance))[0]
					
					if (t_key == 3 or t_key == 13 or t_key == 14) and (ce == 2 or ce == 3):
						print("Exit", " Watchbox: ", events[1], " Detected Track: ", t_key, " Original Track: ", ce, np.array(track_dict[t_key]["frame"])[b_array], " Reference: ", events[3])
					
					if b_array.size > 0:
						b_idx = b_array[0]

						bboxes = np.array(track_dict[t_key]["bbox"])[b_array][:2]
						if len(bboxes.shape) == 1:
							bboxes = np.expand_dims(bboxes,0)
						check[-1] = check[-1] and (not all(FindPoints(watchbox[camera][events[1]],bboxes)))
					else:
						check[-1] = False

		detected_class = np.bincount(track_dict[t_key]["class"]).argmax()
		
		if int(ce) < 4:
			ae_class = 1
		else:
			ae_class = 0
			
		if check and any(check):
			txt_str = "Class: "
			if detected_class == ae_class:
				txt_str += "True"
				print(t_key, "matches", ce, check, txt_str)
			else:
				txt_str += "False"
			
			
				
					
					
"""

	
					
				
			

