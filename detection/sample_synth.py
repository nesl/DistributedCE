'''
import math
import cv2


lane1 = ((1,630),(1920,576))
lane1 = ((1,645),(1918,592))


speed = 80

fps = 10

def lane_length(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    
llength = lane_length(lane1[0], lane1[1])

steps = int(1/(speed/llength)*fps)

p1 = lane1[0]
p2 = lane1[1]

original_image = cv2.imread('frame_1.jpg')

for i in range(steps+1):
    display_point = (int(p1[0]+i/steps*(p2[0]-p1[0])),int(p1[1]+i/steps*(p2[1]-p1[1])))
    image = original_image.copy()
    print(display_point)
    image = cv2.circle(image, display_point, radius=0, color=(0, 0, 255), thickness=10)
    cv2.imshow('image',image)
    cv2.waitKey(1)
    


from matplotlib import pyplot
import numpy as np
from scipy.optimize import curve_fit
   

# define the true objective function
def objective(x, a, b, c, d, e, f):
 return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f
 
 
data = np.array([[1, 525],[112,518],[195,512],[330,506],[528,497],[762,491],[1114,492],[1384,497],[1619,497],[1920,490]])
x,y = data[:,0], data[:,1]

# curve fit
popt, _ = curve_fit(objective, x, y)
# summarize the parameter values
a, b, c, d, e, f = popt


pyplot.scatter(x,y)
# define a sequence of inputs between the smallest and largest known inputs
x_line = np.arange(min(x), max(x), 1)

# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e, f)

...
# create a line plot for the mapping function
pyplot.plot(x_line, y_line, '--', color='red')


pyplot.show()
'''

import numpy as np
from matplotlib import pyplot as plt
import pdb
import subprocess
import os
import argparse
import cv2
import sys


def get_config(scenario_file):
    f = open(scenario_file)
    lines = f.readlines()
    input_config = []
    
    for l_idx, l in enumerate(lines):
        if not l_idx:
            continue
            
        find_comment_char = l.find('#')
        
        if find_comment_char >= 0:
            l = l[:find_comment_char]
        
        if not l:
            continue
        
        line_split = l.strip().split(',')
        line_config = [int(line_split[0]),[]]
        
        new_line_split = line_split[1:]
        
        for ls in range(0,len(new_line_split),3):
            line_config[1].append([int(new_line_split[ls]), float(new_line_split[ls+1]), float(new_line_split[ls+2])])
            
        input_config.append(line_config)
    
    f.close()
    
    return input_config

#Get class information
def get_class_info(input_config, args, max_file):

    desired_classes = list(set([ic[0] for ic in input_config]))
    class_info = {i:[] for i in desired_classes}
    
   
    for i in range(max_file+1):
        file_name = args.tracking_dir + '/' + str(i)+'.txt'
        if os.path.isfile(file_name):
            f = open(file_name)
            lines = f.readlines()
            for l in lines:
                line_split = l.split(',')
                
                line_class = int(line_split[5])
                if line_class in desired_classes:
                    class_info[line_class] = [int(float(line_split[2])-float(line_split[0])),int(float(line_split[3])-float(line_split[1]))]
                    desired_classes.remove(line_class)
                
        if not desired_classes:
            break
            
    return class_info

#Get trajectory information
def get_track_info(input_config, args,max_file):

    desired_track_id = [ic_sub[0] for ic in input_config for ic_sub in ic[1]]
    data = {i:{'box_center':[]} for i in desired_track_id}
    
    
    for i in range(0, max_file+1, args.skip_files):
        file_name = args.tracking_dir + '/' + str(i)+'.txt'
        if os.path.isfile(file_name):
            f = open(file_name)
            lines = f.readlines()
            for l in lines:
                line_split = l.split(',')
                track_id = int(line_split[4])
                
                if track_id in desired_track_id:
                    #box_center = [int(float(line_split[0]) + (float(line_split[2])-float(line_split[0]))/2),int(float(line_split[1]) + (float(line_split[3])-float(line_split[1]))/2)]
                    box = [int(float(line_split[0])), int(float(line_split[1]))]
                
                    data[track_id]['box_center'].append(box)
                    
    return data
    
def get_track_points(data):

    track_points = {}

    #Get points for each track
    for k in data.keys():
        data_track = np.array(data[k]['box_center'])


        x,y = data_track.T
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])

        track_points[k] = (u,x,y)
        
    return track_points


def get_equally_spaced_points(input_config, track_points, fps):

    synth_data = []
    starting_time = []
    
    #Interpolation
    for ic_idx, ic in enumerate(input_config):
        change_times = [speed_points[1] for speed_points in ic[1]]
        
        if len(change_times) == 1:
            periods = [-1]
        else:
            periods = np.diff(change_times)
            periods = np.append(periods, -1)
        
        t_min = 0
        t_max = 0
        xn = np.array([])
        yn = np.array([])
        
        #u = track_points[ic[0]][0]
        #x = track_points[ic[0]][1]
        #y = track_points[ic[0]][2]
        
        #pdb.set_trace()
        for p_idx, p in enumerate(periods):
        
            speed = ic[1][p_idx][2]
            
            u = track_points[ic[1][p_idx][0]][0]
            x = track_points[ic[1][p_idx][0]][1]
            y = track_points[ic[1][p_idx][0]][2]
            
            
            if p == -1:
                number_points = int(1/(speed/(u.max()-t_max))*fps)
                t_min = t_max
                t_max = u.max()
            else:
                number_points = int(p*fps)
            
                if not p_idx:
                    t_min = 0
                    t_max = speed*p
                else:
                    t_min = t_max
                    t_max += speed*p
            
                
            #number_points = int(1/(speed/u.max())*fps)
            #print(1/(speed/u.max()))

            #t = np.linspace(0,u.max(),number_points)
            t = np.linspace(t_min, t_max, number_points)
            
            if xn.size > 0:
                xn = np.concatenate((xn,np.interp(t, u, x)))
                yn = np.concatenate((yn,np.interp(t, u, y)))
            else:
                xn = np.interp(t, u, x)
                yn = np.interp(t, u, y)
                
        synth_data.append([xn,yn,xn.shape[0],ic[0],ic_idx])
        starting_time.append(int(ic[1][p_idx][1]*fps))
        
    return synth_data, starting_time


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Data Synthesizing')
    parser.add_argument('--save-output-dir', default='', type=str, help='Save output to directory specified')
    parser.add_argument('--scenario-file', default='', type=str, help='File with scripted scenario')
    parser.add_argument('--tracking-dir', type=str, help='Directory with tracking results to associate with specified tracks')
    parser.add_argument('--tracks', default='', type=str, help='Tracks to consider, list separated by commas')
    parser.add_argument('--classes', default='', type=str, help='Classes to consider, list separated by commas')
    parser.add_argument('--num-objects', default=0, type=int, help='Number of objects to simulate')
    parser.add_argument('--display', action='store_true', help='Display animation')
    parser.add_argument('--episode-len', default=30, type=int, help='Episode length in time (s)')
    parser.add_argument('--fps', default=30, type=int, help='FPS')
    parser.add_argument('--skip-files', default=5, type=int, help='Number of files to skip when reading the tracking files')
    parser.add_argument('--display-image', default='frame_1.jpg', type=str, help='Name of image to display with animation')

    # sys.argv includes a list of elements starting with the program
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()

    #Create directory for saving output results
    if args.save_output_dir:
        if not os.path.exists(args.save_output_dir):
            os.makedirs(args.save_output_dir)


    cmd = 'ls ' + args.tracking_dir + '/ | sort -n | tail -n 1'
    out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = out.communicate()[0]


    max_file = int(output[:-5])
    print(max_file)


    #np.random.randint(2, size=args.num_objects)

    #input_config = [[0, [[60,2.5,200],[60,4,300], [60,7,50]]], [0, [[5,3,200]]], [0, [[40, 5, 100]]]]
    input_config = get_config(args.scenario_file)


    episode_length = args.episode_len
    fps = args.fps


    class_info = get_class_info(input_config, args, max_file)


    data = get_track_info(input_config, args,max_file)  
                    

    track_points = get_track_points(data)
        

    synth_data, starting_time = get_equally_spaced_points(input_config, track_points, fps)




    original_image = cv2.imread(args.display_image)


    #Print/save results
    
    for i in range(episode_length*fps):
        image = original_image.copy()
        
        to_delete = []
        create_file = False
        text_output = ''
        
        for s_idx,s in enumerate(starting_time):
            if i >= s:
                if i < s+synth_data[s_idx][2]:
                    display_point = (int(synth_data[s_idx][0][i-s]),int(synth_data[s_idx][1][i-s]))
                    
                    
                    if args.display:
                        #image = cv2.circle(image, display_point, radius=0, color=(0, 0, 255), thickness=10)
                        image = cv2.rectangle(image, display_point, (display_point[0]+class_info[synth_data[s_idx][3]][0],display_point[1]+class_info[synth_data[s_idx][3]][1]), (0, 0, 255), 1)
                    
                    if args.save_output_dir:
                        text_output += "%f,%f,%f,%f,%d,%d\n" % (*display_point,*class_info[synth_data[s_idx][3]], synth_data[s_idx][4], synth_data[s_idx][3])
                        create_file = True
                else:
                    to_delete.append(s_idx)
            for d in to_delete:
                del synth_data[d]
                del starting_time[d]
                
        if args.save_output_dir and create_file:
             f = open(args.save_output_dir + '/' + str(i) + '.txt', "w")
             f.write(text_output)
             f.close()
                
        if args.display:
            cv2.imshow('image',image)
            cv2.waitKey(1)


