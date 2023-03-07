import json
import subprocess
import os

def get_metadata(scenario_file):
    f = open(scenario_file)
    lines = f.readlines()
    input_config = {}
    tracking_dirs = []

    for l_idx, l in enumerate(lines):
        if not l_idx:
            continue

        find_comment_char = l.find('#')

        if find_comment_char >= 0:
            l = l[:find_comment_char]

        if not l:
            continue

        line_split = l.strip().split(',')

        if line_split[0] not in input_config:
            input_config[line_split[0]] = {}
        if line_split[3] not in input_config[line_split[0]]:
            input_config[line_split[0]][line_split[3]] = []
        input_config[line_split[0]][line_split[3]].append([int(line_split[1]),int(line_split[2])])



    f.close()

    return input_config



#Get trajectory information
def get_track_points(input_config, tracking_dir, desired_classes, max_file):

    desired_track_id = [ic[1] for ic in input_config[tracking_dir]]
    alias_id = {ic[1]:ic[0] for ic in input_config[tracking_dir]}
    data = {int(i):[] for i in alias_id.values()}
    class_info = {i:[0,0] for i in desired_classes}
    
    
    for i in range(0, max_file+1):
        file_name = tracking_dir + '/' + str(i)+'.txt'
        if os.path.isfile(file_name):
            f = open(file_name)
            lines = f.readlines()
            for l in lines:
                line_split = l.split(',')
                track_id = int(line_split[4])
                line_class = int(line_split[5])
                
                if track_id in desired_track_id:
                    #box_center = [int(float(line_split[0]) + (float(line_split[2])-float(line_split[0]))/2),int(float(line_split[1]) + (float(line_split[3])-float(line_split[1]))/2)]
                    box = [int(float(line_split[0])), int(float(line_split[1]))]
                
                    data[int(alias_id[track_id])].append(box)
                    
                if line_class in desired_classes:
                    class_info[int(line_class)][0] = (class_info[line_class][0] + int(float(line_split[2])-float(line_split[0])))/2
                    class_info[int(line_class)][1] = (class_info[line_class][1] + int(float(line_split[3])-float(line_split[1])))/2

                    
    return data, class_info
    
    

metadata_f = get_metadata("tracks_metadata.txt")

track_points_f = open("track_points.json", "w")
track_classes_f = open("track_classes.json", "w")

root_dict_points = {}
root_dict_classes = {}

for key in metadata_f.keys():

    points = {}
    class_info = {}

    for tracking_dir in metadata_f[key]:

        cmd = 'ls ' + tracking_dir + '/ | sort -n | tail -n 1'
        out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = out.communicate()[0]


        max_file = int(output[:-5])
        
        desired_classes = [0,1,2]
        
        
        points_tmp,class_info_tmp = get_track_points(metadata_f[key], tracking_dir, desired_classes, max_file)
        
        points.update(points_tmp)
        
        if not class_info:
            class_info = class_info_tmp
    
    root_dict_points[key] = points
    root_dict_classes[key] = class_info
    
    
    
    '''
    
    for track_id in points.keys():
        points_str = str(metadata_f[key][0][0]) + ',' + str(track_id) + ',' + ','.join(points[track_id]) + '\n'
        track_points_f.write(points_str)
        
    for class_id in class_info.keys():
        class_str = str(metadata_f[key][0][0]) + ',' + class_info
    '''
    
    
track_points_f.write(json.dumps(root_dict_points))
track_classes_f.write(json.dumps(root_dict_classes))
    
