import pybboxes as pbx
import cv2
import pdb
import argparse
from transformers import pipeline
from PIL import Image
import os
from yolov5.detect_simple import Yolo_Exec
import torch
from torchvision import ops
from trackers.tracker.byte_tracker import BYTETracker
import numpy as np

#Run yolo model, extract crops classify


def sorting_function(e):
	return float(e.split()[5])
	
def sorting_labels(e):
	return int(e["label"][-1:])

parser = argparse.ArgumentParser(description='Showing BBs for frame in video')
parser.add_argument('--vid-name', type=str, help='Video file')
parser.add_argument('--start', type=int, default=0, help='Frame Number')
parser.add_argument('--end', type=int, default=-1, help='Frame Number')
parser.add_argument('--bbox-folder', type=str, help='Directory with BBox results')
parser.add_argument('--write', action="store_true", help = "Save crop images")
parser.add_argument('--interactive', '-i', action="store_true", help = "Display images as slideshow. Move with w and s, q to quit")
parser.add_argument('--model', type=str, default='', help='Model directory')
parser.add_argument('--watchbox', type=str, default='', help='Watchbox in the form of "[x1,y1,x2,y2]"')
parser.add_argument('--conf-thres', type=float, default=0.1, help='YOLO confidence threshold')
parser.add_argument('--yolo-weights', nargs='+', type=str, default='', help="YOLO weights file")
parser.add_argument('--device', type=str, default='0', help="Device where to run YOLO")
parser.add_argument('--labels', type=str, default='', help="Labels to check as a comma-separated list")
parser.add_argument('--write-crop-result', type=str, default='', help="Directory to save the crop results")
parser.add_argument('--create-video', type=str, default='', help='Create video. Specify file name.')
parser.add_argument('--display', action='store_true', help='Display video')
parser.add_argument('--get-gt', action='store_true', help='Get ground truth predictions')
parser.add_argument('--write-gt-dir', type=str, default='', help='Write ground truth to a directory.')

args = parser.parse_args()

labels = []

if args.labels:
	labels = args.labels.split(",")

vid_name = args.vid_name
bbox_folder = args.bbox_folder
frame_num = args.start-1 #For start


if args.watchbox:
	watchbox = eval(args.watchbox)

cap = cv2.VideoCapture(vid_name)

pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fontScale = 0.5
color = (255, 153, 255)
# Line thickness of 2 px
thickness = 2
font = cv2.FONT_HERSHEY_SIMPLEX

			
			



class ByteTrackArgs:
    def __init__(self, track_thresh, match_thresh, track_buffer, low_track_thres):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.mot20 = False 
        self.track_buffer = track_buffer
        self.low_track_thres = low_track_thres

new_args = ByteTrackArgs(args.conf_thres,0.99, 100, args.conf_thres)#20)
tracker = BYTETracker(new_args)
tracker.args.mot20 = True

increment = 1


if args.write_crop_result:
	metadata_f = open(args.write_crop_result + "/metadata.csv", "w")
	
if args.write_gt_dir:
	write_gt_dir = args.write_gt_dir + '/'
	if os.path.exists(write_gt_dir):
		os.system("rm -r " + write_gt_dir)    
	os.makedirs(write_gt_dir)


if args.yolo_weights:
	yolo = []
	for y in args.yolo_weights:
		yolo.append(Yolo_Exec(weights=y, imgsz=[pixel_width],conf_thres=args.conf_thres, device=args.device, save_conf=True))
	
	if args.get_gt:
		soartech_yolo = Yolo_Exec(weights="yolov5_original/soartechDetectorV2.pt", imgsz=[pixel_width],conf_thres=0.1, device=args.device, save_conf=True)


if args.create_video:
	new_video = cv2.VideoWriter(args.create_video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (pixel_width,pixel_height))


bbox_history = {}
lines_dict = {}

while True:

	frame_num += increment
	
	

	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

	ret,frame = cap.read()


	frames = []
	crops = []
	boxes = []
	confidences = []

	if ret:
	
		if args.yolo_weights:
			res_lines = []
			for y in yolo:
				res_lines.extend(y.run(frame))
			
			if args.get_gt:
				soartech_res_lines = soartech_yolo.run(frame)
				soartech_remain = []
				for line in soartech_res_lines:
					current_label = line.split()[0]
					if current_label not in labels:
						soartech_remain.append(' '.join(line.split()[:5])+'\n')
				lines_dict[frame_num] = soartech_remain
					
			if not res_lines:
				continue
		else:
			file_bbox_name = bbox_folder+str(frame_num)+".txt"
	
			if not os.path.exists(file_bbox_name):
				continue
				
			
			
			file_bbox = open(file_bbox_name)

			yolo_out = file_bbox.readlines()
			res_lines = list(map(lambda s: s.strip(), yolo_out))
			
		res_lines.sort(key=sorting_function, reverse=True)
		
		print("Frame: ", frame_num)

		for line_idx,line in enumerate(res_lines):
			coordinates_line = line.split()
			
			frame_copy = frame.copy()

			box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),float(coordinates_line[4])), from_type="yolo", to_type="voc", image_size=(pixel_width,pixel_height))
			
			
			if args.watchbox:
				if not (box_voc[0] > watchbox[0] and box_voc[1] > watchbox[1] and box_voc[2] < watchbox[2] and box_voc[3] < watchbox[3]):
					continue
			
			boxes.append(box_voc)
			confidences.append(float(coordinates_line[5]))
			
			cv2.rectangle(frame_copy, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)
			
			frame_copy = cv2.putText(frame_copy, coordinates_line[0]+" "+coordinates_line[5]+" "+coordinates_line[6], (box_voc[0], box_voc[1]), font, fontScale, color, thickness, cv2.LINE_AA)
			
			frames.append(frame_copy)

			crop = frame[box_voc[1]:box_voc[3], box_voc[0]:box_voc[2]]
			
			crops.append(crop)
			
			if args.write:
				cv2.imwrite("crop_" + str(line_idx) + ".jpg", crop)
			
			#cv2.imshow("Frame_"+str(line_idx), frame_copy)
			#cv2.waitKey(100)
			#cv2.destroyAllWindows()
			"""
			while True:
				if cv2.waitKey(0) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
			"""	
		
		all_frame = frame.copy()
		for box_voc in boxes:
			cv2.rectangle(all_frame, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)
		
	else:
		print("No image")
		frame_num = -1
		
		if args.interactive:
			continue
		else:
			break
		

		

	if args.model:
		classifier = pipeline("image-classification", model=args.model, device=0)

	if not frames:
		print("No detections")
		continue
		
	if args.interactive:
		index = 0
		show_all = False
		while True:
		
			if not frames:
				cv2.imshow("Frame", frame)
			else:
				if not show_all:
					cv2.imshow("Frame", frames[index])
				else:
					cv2.imshow("Frame", all_frame)
				
			key = cv2.waitKey(10) & 0xFF
			
			if key == ord('w'):
				index = (index + 1) % len(frames)
			elif key == ord('s'):
				index = (index - 1) % len(frames)
			elif key == ord('n'):
				increment = 1
				break
			elif key == ord('p'):
				increment = -1
				break
			elif key == ord('a'):
				show_all = not show_all
			elif key == ord('q'):
				quit()
			elif args.model and key == ord('e') and frames:
			
				img = cv2.cvtColor(crops[index], cv2.COLOR_BGR2RGB)
				im_pil = Image.fromarray(img)
				results = classifier(im_pil)
				print(results)
	elif args.end != -1 and frame_num >= args.end:
		break
	else:
		ious = ops.box_iou(torch.tensor(boxes), torch.tensor(boxes))
		to_delete = []
		to_maintain = []
		for iou_idx,iou in enumerate(ious):
			if iou_idx not in to_delete:
				to_maintain.append(iou_idx)
				for col_idx,col in enumerate(iou):
					if col_idx == iou_idx:
						continue
					if col > 0.45:
						to_delete.append(col_idx)
						
		txt_line = ""
		
		detection_bboxes = np.array([])
		detection_class_ids = np.array([])
		detection_confidences = np.array([])
		detection_extra = np.array([])
		frame_copy = frame.copy()
		
		im_pil = []
		for c_idx in to_maintain:
			img = cv2.cvtColor(crops[c_idx], cv2.COLOR_BGR2RGB)
			im_pil.append(Image.fromarray(img))
		
		results = classifier(im_pil, top_k=6)

		for r_idx,c_idx in enumerate(to_maintain):
			#print(results[0], boxes[c_idx])

			class_label = results[r_idx][0]["label"][-1:]
			#if labels and class_label in labels and results[0]["score"] > 0.5:
			
			#if results[0]["score"] > 0.5:
			
			if detection_bboxes.size == 0:
				detection_bboxes = np.expand_dims(np.array(boxes[c_idx]),axis=0)
			else:
				detection_bboxes = np.concatenate((detection_bboxes, np.expand_dims(np.array(boxes[c_idx]),axis=0)),axis=0)
			#detection_bboxes = np.append(detection_bboxes, np.expand_dims(np.array(box_voc),axis=0),axis=0)
			detection_class_ids = np.append(detection_class_ids, int(class_label))
			detection_confidences = np.append(detection_confidences, confidences[c_idx])

			results[r_idx].sort(key=sorting_labels)
			extra_data = np.array([cl["score"] for cl in results[r_idx]])
			if detection_extra.size == 0:
				detection_extra = np.expand_dims(extra_data,axis=0)
			else:
				detection_extra = np.concatenate((detection_extra,np.expand_dims(extra_data,axis=0)),axis=0 )
			
			cv2.rectangle(frame_copy, (int(boxes[c_idx][0]), int(boxes[c_idx][1])), (int(boxes[c_idx][2]), int(boxes[c_idx][3])), (255, 0, 0), 1)
		
			#yolo_box = pbx.convert_bbox(tuple(boxes[c_idx]), from_type="voc", to_type="yolo", image_size=(pixel_width,pixel_height))
			#txt_line += "%d %f %f %f %f\n" % (int(class_label), *yolo_box) 
			if args.write_crop_result:
				filename_crop = args.write_crop_result + '/' + str(frame_num) + "_" + str(c_idx) + ".jpg"
				cv2.imwrite(filename_crop, crops[c_idx])
				metadata_f.write(filename_crop + "," + class_label + "\n")
			
		
		bbox_stack = np.column_stack((detection_bboxes,detection_confidences))

		#pdb.set_trace()
		online_targets,lost_tracks,removed_tracks = tracker.update(bbox_stack, [pixel_height, pixel_width], (pixel_height,pixel_width), detection_class_ids, detection_extra) #check order of data error
		
		online_targets_len = len(online_targets)
		
		
		
		image = cv2.putText(frame_copy, ' Frame: ' + str(frame_num), (50, 50), font, 
	           fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
		for t_idx,t in enumerate([*online_targets,*lost_tracks]):

			bbox = t.tlbr
			
			if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
				continue
				
			for bb_idx in range(len(bbox)):
				if bbox[bb_idx] < 0:
					bbox[bb_idx] = 0
				if bb_idx in [0,2] and bbox[bb_idx] > pixel_width:
					bbox[bb_idx] = pixel_width
				if bb_idx in [1,3] and bbox[bb_idx] > pixel_height:
					bbox[bb_idx] = pixel_height
					
			if abs(int(bbox[0]) - int(bbox[2])) < 5 or abs(int(bbox[1]) - int(bbox[3])) < 5:
				continue

			if args.watchbox:
				if not (bbox[0] > watchbox[0] and bbox[1] > watchbox[1] and bbox[2] < watchbox[2] and bbox[3] < watchbox[3]):
					continue

			track_id = t.track_id
			
			if track_id not in bbox_history:
				bbox_history[track_id] = {"frame":[],"bbox":[]}
				
			bbox_history[track_id]["bbox"].append(bbox)
			bbox_history[track_id]["frame"].append(frame_num)
			
			class_history = t.detected_class #For Bytetracker, haven't tested with other ones
			detection_extra = t.detected_extra		
			class_detected = np.bincount(class_history).argmax()

			cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
			
			if t_idx < online_targets_len:
				color = (255, 153, 255) #violet
			else:
				color = (255, 255, 0) #cyan
			
			frame_copy = cv2.putText(frame_copy, str(track_id) + "_" + str(class_detected), (int(bbox[0]), int(bbox[1])), font, fontScale, color, thickness, cv2.LINE_AA)			

			try:
				yolo_box = pbx.convert_bbox(tuple(bbox), from_type="voc", to_type="yolo", image_size=(pixel_width,pixel_height))
			except:
				pdb.set_trace()

			txt_line += "%d %f %f %f %f %d\n" % (class_detected, *yolo_box, track_id) 
			
		#print(txt_line)
		
		if args.watchbox:
			cv2.rectangle(frame_copy, (watchbox[0], watchbox[1]), (watchbox[2], watchbox[3]), (0, 0, 0), 1)
		
		cv2.imshow("Frame", frame_copy)
				
		cv2.waitKey(1)
			
		if args.create_video:
			new_video.write(frame_copy)
	
#frame_copy = frame.copy()
colors = [(0, 0, 255),(0, 0, 0),(255, 255, 255),(0, 255, 0),(255, 0, 0),(0, 255, 255),(255, 0, 255),(255, 255, 0),(0, 128, 255),(128, 0, 255),(255, 128, 0),(255, 0, 128),(128, 128, 128),(0, 255, 128),(128, 255, 0),(0, 0, 128),(0, 128, 0),(128, 0, 0),(0, 128, 128),(128, 0, 128),(128, 128, 0)]

color_idx = 0

tracks_processed = []

for t_idx,track in enumerate([*online_targets,*lost_tracks,*removed_tracks]):
	class_detected = np.bincount(track.detected_class).argmax()
	#if str(class_detected) in labels:
	if track.track_id in bbox_history.keys() and track.track_id not in tracks_processed:
	
		tracks_processed.append(track.track_id)
		
		for b_idx in range(len(bbox_history[track.track_id]["frame"])):
			yolo_box = pbx.convert_bbox(tuple(bbox_history[track.track_id]["bbox"][b_idx]), from_type="voc", to_type="yolo", image_size=(pixel_width,pixel_height))
			
			dict_key = bbox_history[track.track_id]["frame"][b_idx]
			if dict_key not in lines_dict:
				lines_dict[dict_key] = []
				
			lines_dict[dict_key].append("%d %f %f %f %f %d\n" % (class_detected, *yolo_box, track.track_id))

			#cv2.rectangle(frame_copy, (int(bbox_history[track.track_id]["bbox"][b_idx][0]), int(bbox_history[track.track_id]["bbox"][b_idx][1])), (int(bbox_history[track.track_id]["bbox"][b_idx][2]), int(bbox_history[track.track_id]["bbox"][b_idx][3])), colors[color_idx], 1)
		
		#frame_copy = cv2.putText(frame_copy, str(track.track_id) + "_" + str(class_detected), (int(bbox_history[track.track_id]["bbox"][b_idx][0]), int(bbox_history[track.track_id]["bbox"][b_idx][1])), font, fontScale, colors[color_idx], thickness, cv2.LINE_AA)	
		
		color_idx = (color_idx+1) % len(colors)
		
		

image_tubes = "tubes.jpg"	

print(lines_dict)

if args.write_gt_dir:
	image_tubes = write_gt_dir + image_tubes
	for key in lines_dict.keys():
	
		if lines_dict[key]:
			frame_file = open(write_gt_dir + str(key)+".txt","w")
			frame_file.writelines(lines_dict[key])

#cv2.imwrite(image_tubes, frame_copy)
	
if args.create_video:
	new_video.release()
