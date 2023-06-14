import pybboxes as pbx
import cv2
import pdb
import argparse
import glob
import os
import random
import torch
from torchvision import ops

#Extract crops from files with multiple detections

parser = argparse.ArgumentParser(description='Showing BBs for frame in video')
#parser.add_argument('--vid-name', type=str, help='Video file')
#parser.add_argument('--frame', type=int, help='Frame Number')
parser.add_argument('--image-folder', type=str, help='Directory with Image results')
parser.add_argument('--bbox-folder', type=str, help='Directory with BBox results')
parser.add_argument('--crop-folder', type=str, help='Directory with crop results')

args = parser.parse_args()


bbox_folder = args.bbox_folder + "/"
image_folder = args.image_folder + "/"
crop_folder = args.crop_folder + "/"

if os.path.exists(crop_folder):
	os.system("rm -r " + crop_folder)
os.mkdir(crop_folder)


bbox_files = glob.glob(bbox_folder + '*.txt')

metadata_f = open(crop_folder + "metadata.csv", "w")
metadata_f.write("image,label\n")

for bbox_idx,bbox_f in enumerate(bbox_files):

	open_file = open(bbox_f)
	yolo_out = open_file.readlines()
	res_lines = list(map(lambda s: s.strip(), yolo_out))

	base_bbox = os.path.basename(bbox_f)
	original_image = cv2.imread(image_folder+base_bbox[:-4]+".jpg")

	pixel_width = original_image.shape[1]
	pixel_height = original_image.shape[0]
	
	bboxes = []
	
	for line_idx,line in enumerate(res_lines):
	
		image = original_image.copy()
	
		coordinates_line = line.split()	

		old_box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),float(coordinates_line[4])), from_type="yolo", to_type="voc", image_size=(pixel_width,pixel_height))
		
		box_voc = []
		for bb_idx,bb in enumerate(old_box_voc):
			if bb < 0:
				bb = 0
			if bb_idx in [0,2] and bb > pixel_width:
				bb = pixel_width
			if bb_idx in [1,3] and bb > pixel_height:
				bb = pixel_height
			box_voc.append(bb)
			
		if abs(box_voc[0] - box_voc[2]) < 1 or abs(box_voc[1] - box_voc[3]) < 1:
			continue
		
		bboxes.append(box_voc)
		crop = image[box_voc[1]:box_voc[3], box_voc[0]:box_voc[2]]
		
		new_file_name = crop_folder + str(bbox_idx) + '_' + str(line_idx) + ".jpg"
		
		metadata_f.write(new_file_name + "," + coordinates_line[0] + '\n')
		
		
		cv2.imwrite(new_file_name, crop)

		
		#cv2.imshow("Frame_", crop)
		
		#cv2.waitKey(1000)


	if not bboxes:
		continue
		
	image = original_image.copy()
	
	valid_env_bbox = False
	
	while not valid_env_bbox:
	
		x1 = random.randint(0,pixel_width-20)
		y1 = random.randint(0,pixel_height-20)
		
		box_voc = [x1,y1,random.randint(x1+10,pixel_width),random.randint(y1+10,pixel_height)]
		
		ious = ops.box_iou(torch.tensor(bboxes), torch.tensor([box_voc]))
		
		valid_env_bbox = torch.all(torch.where(ious == 0, 1, 0))
		
	crop = image[box_voc[1]:box_voc[3], box_voc[0]:box_voc[2]]
	new_file_name = crop_folder + str(bbox_idx) + '_' + str(line_idx+1) + ".jpg"
	
	metadata_f.write(new_file_name + ",5\n")
	

	cv2.imwrite(new_file_name, crop)


metadata_f.close()
