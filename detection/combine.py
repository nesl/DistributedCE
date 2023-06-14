import os

#combine detections of yolo with ground truth files

list_of_videos = os.popen('ls "/home/nesl/Projects/complex_event/special/DistributedCE_testing/detection/yolov5/frames/" | cut -d "_" -f 1 | sort | uniq').read().split('\n')[:-1]

original_labels_path = "/home/nesl/Projects/complex_event/special/DistributedCE_testing/detection/yolov5/labels/"

labels_path = "results_smoke/"

new_path = "combined_labels/"

if os.path.exists(new_path):
    os.system("rm -r " + new_path)
    
os.makedirs(new_path)

dir_names = os.listdir(labels_path)

for n in range(len(list_of_videos)):
    if any(list_of_videos[n] == d.split("_")[0] for d in dir_names):
        take_dir = labels_path+list_of_videos[n]+"_"
        
        for camera in [2,3]:
            camera_dir = take_dir+str(camera)+"/"
            if os.path.exists(camera_dir):
                for label_file in os.listdir(camera_dir):
                    #file_pattern = list_of_videos[n]+"_"+str(camera)+"_"+label_file
                    original_file = original_labels_path+list_of_videos[n]+"_"+str(camera)+"_"+str(int(label_file[:-4])+23)+'.txt'
                    
                    complete_yolo_out = open(camera_dir+label_file).readlines()
                    
                    
                    if os.path.exists(original_file):
                        original_yolo_out = open(original_file).readlines()
                        
                        for ol in complete_yolo_out:
                            if int(ol.split()[0]) >= 2:
                                original_yolo_out.append(ol)
                        #else:
                        #    original_yolo_out = complete_yolo_out
                                
                        new_file = open(new_path+list_of_videos[n]+"_"+str(camera)+"_"+label_file, "w")
                    
                        new_file.writelines(original_yolo_out)
                        new_file.close()
                        
        

