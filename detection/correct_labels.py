import pickle
import re
import os
import glob
import math
import pdb

#Correct labels with offset

ground_truth = "/home/nesl/Projects/complex_event/special/DistributedCE_testing/detection/ground_truth_csv/"

f =  open(ground_truth + "offsets.pkl", "rb")
offset_data = pickle.load(f)

frame_path = "/home/nesl/Projects/complex_event/special/DistributedCE_testing/detection/yolov5/frames/"
labels_path = "/home/nesl/Projects/complex_event/special/DistributedCE_testing/detection/yolov5/labels/"

destination_labels = "/home/nesl/Projects/complex_event/special/DistributedCE_testing/detection/ground_truth_csv/labels/"
destination_frames = "/home/nesl/Projects/complex_event/special/DistributedCE_testing/detection/ground_truth_csv/frames/"

bbox_files = glob.glob(ground_truth + '*.csv')

cam_num_csv = re.compile(r'NAI (\d+)')
fps = 30


def create_symlink(target,link_name):
	if os.path.islink(link_name):
		os.remove(link_name)
	os.symlink(target, link_name)


for b in bbox_files:

	bbase = os.path.basename(b)
	take = bbase.split("_")[1]
	
	offset = offset_data["take_"+take]
	
	ground_truth_file = open(b)
	ground_truth_file.readline()
        
	while True:
        
		line = ground_truth_file.readline()
        	
		if not line:
			break
        		
		line = line.strip().split(',')
        		
		mo = cam_num_csv.search(line[3])
	        
		if mo:
			real_offset = int(line[2]) - offset + 3
			camera = mo.group(1)
			
			frame_id = math.trunc(float(line[1])*fps)
			
			file_template = take+"_"+camera+"_"
			
			source_file = file_template + str(frame_id)
			dest_file = file_template + str(real_offset)
			
			try:
				create_symlink(labels_path+source_file+".txt", destination_labels+dest_file+".txt")
				create_symlink(frame_path+source_file+".jpg", destination_frames+dest_file+".jpg")
			except:
				pdb.set_trace()
				print("hekki")
        
	
	
	
	
	
	
	
	
	
	
