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
import json


waypoints_index = 2
num_elements_waypoint = 5

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
        line_config = [int(line_split[0]),int(line_split[1]),[]]
        
        new_line_split = line_split[waypoints_index:]
        
        for ls in range(0,len(new_line_split),num_elements_waypoint):
            line_config[2].append([int(new_line_split[ls]), float(new_line_split[ls+1]), float(new_line_split[ls+2]), float(new_line_split[ls+3]), float(new_line_split[ls+4])])
            
        input_config.append(line_config)
    
    f.close()
    
    return input_config

#Get class information
def get_class_info(input_config, classes, desired_classes):

    #desired_classes = list(set([ic["class"] for ic in input_config]))
    class_info = {i:{"box":[],"image":[]} for i in desired_classes}
    

        
        
    for line_class in desired_classes:
        class_info[line_class]["box"] = classes[str(line_class)]
        #image_tmp = cv2.imread("crops/"+str(line_class)+'.jpg')
        image_tmp = cv2.imread("crops/"+str(line_class)+'.png',cv2.IMREAD_UNCHANGED)


        image_tmp = cv2.resize(image_tmp, (int(class_info[line_class]["box"][0]), int(class_info[line_class]["box"][1])), interpolation = cv2.INTER_AREA)
        class_info[line_class]["image"] = image_tmp
        
    
    '''
    for i in range(max_file+1):
        file_name = tracking_dir + '/' + str(i)+'.txt'
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
    '''
      
    return class_info

#Get trajectory information
def get_track_points(input_config, skip_files, points, desired_track_id):

    #desired_track_id = [ic["track"] for ic in input_config]
    data = {i:{'normal':[], 'reverse':[]} for i in desired_track_id}
    
    for track_id in desired_track_id:
        data[track_id]['normal'] = points[str(track_id)][::skip_files]
        data[track_id]['reverse'] = list(reversed(points[str(track_id)][::skip_files]))
    

    '''
    for i in range(0, max_file+1, skip_files):
        file_name = tracking_dir + '/' + str(i)+'.txt'
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
    '''             
    return data
    
def process_points(data):

    track_points = {}
   

    #Get points for each track
    for k in data.keys():
        for order in data[k].keys():
            data_track = np.array(data[k][order])


            x,y = data_track.T
            xd = np.diff(x)
            yd = np.diff(y)
            dist = np.sqrt(xd**2+yd**2)
            u = np.cumsum(dist)
            u = np.hstack([[0],u])

            if k not in track_points:
                track_points[k] = {}    
            track_points[k][order] = (u,x,y)
        
    return track_points


def check_constraints(synth_data, t_temp, x_temp, y_temp, ic_track, episode_len_fps, vehicle_separation):

    #if max(t_temp) > episode_len_fps:
    #    return 1

    for sd in synth_data:
        if sd[5] == ic_track:
            intersc, x_ind, y_ind = np.intersect1d(t_temp, sd[4], return_indices=True)                    
            if intersc.size > 0:
                for i in range(len(intersc)):
                    res = np.linalg.norm(np.array([x_temp[x_ind[i]],y_temp[x_ind[i]]]) - np.array([sd[0][y_ind[i]],sd[1][y_ind[i]]]))
                    if res < vehicle_separation:
                        return 1
                        
    return 0

def get_equally_spaced_points(input_config, track_points, fps, episode_len, extra_vehicles):

    synth_data = []
    starting_time = []
    traffic_points = []
    vehicle_separation = 50
    
    orders = ["reverse" if "reverse" in ic and int(ic["reverse"]) else "normal" for ic in input_config]
    
    num_vehicles = len(input_config) + extra_vehicles
     
    
    #Interpolation
    #for ic_idx, ic in enumerate(input_config):
    for vehicle_idx in range(num_vehicles):
        
    
        if vehicle_idx >= len(input_config):


            possible_classes = [0,2]
            possible_orders = ["normal", "reverse"]
            track_orders = [1,0,1,0]
            
            ic_class = possible_classes[int(np.random.randint(2, size=1))]

                        
            
            ic_track = int(np.random.randint(4,size=1))
            ic_order = possible_orders[track_orders[ic_track]]
            
        else:
            ic = input_config[vehicle_idx]
            ic_class = ic["class"]
            ic_trajectory = ic["trajectory"]
            ic_track = ic["track"]
            if "reverse" in ic and int(ic["reverse"]):
                ic_order = "reverse"
            else:
                ic_order = "normal"

        
        
        solved = False
        while not solved:
        
            solved = True
            stopped = -1
            constraints_checked = False

            if vehicle_idx >= len(input_config):
                ic_trajectory = [{}]
                ic_trajectory[0]["type"] = "time"
                ic_trajectory[0]["start"] = np.random.rand(1)[0]*episode_len
                ic_trajectory[0]["speed"] = np.random.normal(200, 10, 1)[0]
                
                for tp_idx in range(len(traffic_points)):
                    if ic_trajectory[0]["start"]*fps >= traffic_points[tp_idx][2] and ic_track == traffic_points[tp_idx][3]:
                        ic_order = "normal"
                        ic_trajectory.append({"type":"location", "start":[traffic_points[tp_idx][0],traffic_points[tp_idx][1]] ,"speed": 0, "separation": traffic_points[tp_idx][4]})

                        stopped = tp_idx
                        #constraints_checked = True
                        break

            
            
        

            
            
            
        
            change_points = [-1 if speed_points["type"] == "location" else speed_points["start"] for speed_points in ic_trajectory]
            type_of_change = [speed_points["type"] for speed_points in ic_trajectory]
            speeds = [speed_points["speed"] for speed_points in ic_trajectory]
            
            reverses = []
            
            t_min = 0
            t_max = 0
            xn = np.array([])
            yn = np.array([])
            
            #u = track_points[ic[0]][0]
            #x = track_points[ic[0]][1]
            #y = track_points[ic[0]][2]
            
            #pdb.set_trace()

            for p_idx, p in enumerate(change_points):
            
                speed = speeds[p_idx]
                
               
                u = track_points[ic_track][ic_order][0]
                x = track_points[ic_track][ic_order][1]
                y = track_points[ic_track][ic_order][2]
                
                if "traffic" in ic_trajectory[p_idx]: #This is a point to generate traffic jam
                    traffic_points.append([xn[-1],yn[-1],tn[-1], ic_track, vehicle_separation])


                
                #Get number of points based on speed
                if speed < 0:
                    number_points = abs(int(1/(speed/(t_max-0))*fps))
                elif speed == 0: #If the vehicle stops
                    if p_idx < len(change_points)-1:
                        if not type_of_change[p_idx+1] == "time":
                            print("Error: When stopped, you need to specify a time for the next action")
                            return [],[]
                        period = change_points[p_idx+1] - change_points[p_idx]
                        
                    else:
                        period = episode_len - change_points[p_idx]
                        
                    period *= fps
                    
                    if period < 0: #If period negative it means change is after episode ends
                        solved = False
                        break

                    x_temp = np.full((int(period),), xn[-1])
                    y_temp = np.full((int(period),), yn[-1])
               
                    xn = np.concatenate((xn,x_temp))

                    yn = np.concatenate((yn,y_temp))
                    t_temp = np.arange(tn[-1]+1,tn[-1]+1+len(x_temp))
                    tn = np.concatenate((tn,t_temp))
                    continue
                else:
                    number_points = int(1/(speed/(u.max()-t_max))*fps)
                
                

                if p_idx < len(change_points)-1: #If there are still more trajectory parts
                
                    if type_of_change[p_idx+1] == "location" or type_of_change[p_idx+1] == "location_range": #Particular point

                        if speed < 0:
                            t = np.linspace(t_max, 0, number_points)
                        else:
                            t = np.linspace(t_max, u.max(), number_points)
                            
                        #Get rest of trajectory
                        xs = np.interp(t, u, x)
                        ys = np.interp(t, u, y)
                        
                        min_value = -1
                        min_index = -1
                        
                        extra_distance = 0
                        
                        if type_of_change[p_idx+1] == "location":
                            x_location = ic["trajectory"][p_idx+1]["start"][0]
                            y_location = ic["trajectory"][p_idx+1]["start"][1]
                            
                            if "separation" in ic_trajectory[p_idx+1]:
                                extra_distance = ic_trajectory[p_idx+1]["separation"]

                            
                        elif type_of_change[p_idx+1] == "location_range":
                            x_points = np.sort([ic["trajectory"][p_idx+1]["start"][0],ic["trajectory"][p_idx+1]["end"][0]])
                            y_points = np.sort([ic["trajectory"][p_idx+1]["start"][1],ic["trajectory"][p_idx+1]["end"][1]])


                            new_loc = np.random.uniform(low=[x_points[0],y_points[0]], high=[x_points[1],y_points[1]], size=(1,2))
                            x_location = new_loc[0][0]
                            y_location = new_loc[0][1]
                        
                        #Get the point that is closer to the location defined by the user
                        for c_idx in range(len(xs)):

                            dist = np.linalg.norm(np.array([xs[c_idx],ys[c_idx]])-np.array([x_location,y_location])) #location
                            
                            if min_value == -1 or (min_value > dist and dist > extra_distance):
                                min_value = dist
                                min_index = c_idx
                                
                        xd = np.diff(xs)
                        yd = np.diff(ys)
                        new_dist = np.sqrt(xd**2+yd**2)
                        new_u = np.cumsum(new_dist)
                        new_u = np.insert(new_u, 0,0)
                        total_distance = new_u[min_index] #Get distances to get the point where the vehicle should change
                        change_points[p_idx+1] = change_points[p_idx]+total_distance/abs(speed)
                        
                        if min_index == 0: #Starting position is outside image
                            solved = False
                            break

                        
                    elif type_of_change[p_idx+1] == "distance": #Only distance
                        change_points[p_idx+1] = change_points[p_idx]+change_points[p_idx+1]/abs(speed)
                        
                    period = change_points[p_idx+1] - change_points[p_idx]
                    
                
                else:
                    period = -1
                
                    

                #If it's not the last  part of the trajectory
                if period > -1:
                    number_points = int(period*fps)
                
                    if not p_idx:
                        t_min = 0
                        t_max = speed*period
                    else:
                        t_min = t_max
                        t_max += speed*period
                else:
                    t_min = t_max
                    if speed < 0:
                        t_max = 0
                    else:
                        t_max = u.max()
                
                #print(t_min, t_max, period, speed, number_points)
                #number_points = int(1/(speed/u.max())*fps)
                #print(1/(speed/u.max()))

                #t = np.linspace(0,u.max(),number_points)

                #Make interpolation
                t = np.linspace(t_min, t_max, number_points)

                
                x_temp = np.interp(t, u, x)
                y_temp = np.interp(t, u, y)
                
                
                #Compute lateral offset
                if "offset" in ic["trajectory"][p_idx]:

                    offset = ic["trajectory"][p_idx]["offset"]
                    xsum = []
                    ysum = []
                    slope = []
                    for c_idx in range(1,len(x_temp)):
                        slope = (y_temp[c_idx]-y_temp[c_idx-1])/(x_temp[c_idx]-x_temp[c_idx-1])
                        b = y_temp[c_idx] - slope*x_temp[c_idx] + offset
                        y_temp[c_idx] = slope*x_temp[c_idx] + b
            
            

            

                
                #Append data points
                if xn.size > 0:
                    xn = np.concatenate((xn,x_temp))
                    yn = np.concatenate((yn,y_temp))
                    t_temp = np.arange(tn[-1]+1,tn[-1]+1+len(x_temp))
                    tn = np.concatenate((tn,t_temp))
                else:
                
                    t_temp = np.arange(int(ic_trajectory[0]["start"]*fps),int(ic_trajectory[0]["start"]*fps)+len(x_temp))
                    
                    
                    if not constraints_checked:
                        #Check constraints
                        res = check_constraints(synth_data, t_temp, x_temp, y_temp, ic_track, episode_len*fps, vehicle_separation)

                        if res:
                            solved = False
                            #print("constraints not followed")
                            if vehicle_idx < len(input_config):
                                print("constraints not followed")
                                quit()
                            else:
                                break
                        else:
                            print(t_temp[-1]/fps)
                            
                
                    xn = x_temp
                    yn = y_temp
                    tn = t_temp
                    
                
                    
                    
                try:
                #Add reverse times    
                    if speed < 0:
                        reverses.append((True,tn[-1]+1))
                    else:
                        reverses.append((False,tn[-1]+1))
                except:
                    pdb.set_trace()
                


                    
                
        synth_data.append([xn,yn,xn.shape[0],ic_class,tn,ic_track,ic_order, reverses])
        starting_time.append(int(ic_trajectory[0]["start"]*fps))
        
        if vehicle_idx >= len(input_config):
            print(ic_trajectory[0]["start"],ic_class, ic_track)
        #print(int(ic[waypoints_index][0][1]*fps), ic[0])
        
        if stopped >= 0: #Make sure other vehicles stop farther away
            traffic_points[stopped][4] += vehicle_separation
            traffic_points[stopped][2] = ic_trajectory[0]["start"]*fps
            print(stopped, traffic_points[stopped][4], traffic_points[stopped][2] )

        
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
    parser.add_argument('--skip-files', default=1, type=int, help='Number of files to skip when reading the tracking files')
    parser.add_argument('--display-image', default='frame_1.jpg', type=str, help='Name of image to display with animation')
    parser.add_argument('--create-video', type=str, default='', help='Create video. Specify file name.')
    parser.add_argument('--extra-vehicles', type=int, default=0, help='Number of extra vehicles to render')

    # sys.argv includes a list of elements starting with the program
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    args = parser.parse_args()
    
    json_f = open(args.scenario_file)
    json_data = json.load(json_f)
    
    json_points = json.load(open("track_points.json"))
    json_camera = json.load(open("track_classes.json"))
    
    synth_data = {}
    starting_time = {}
    original_image = {}
    class_info = {}
    
    video = {}


    class_metadata =  {}
    f = open('class_metadata.txt')
    lines = f.readlines()
    
    for l_idx, l in enumerate(lines):
        if not l_idx:
            continue
        line_split = l.strip().split(',')
        
        cam = str(line_split[0])
        
        if cam not in class_metadata:
            class_metadata[cam] = {}
        class_metadata[cam][str(line_split[1])] = [int(line_split[2]),int(line_split[3])]
        
    
    for cam in json_data["scenario"]:
        

        #Create directory for saving output results
        if args.save_output_dir:
            
            if not os.path.exists(args.save_output_dir):
                os.makedirs(args.save_output_dir)
                
            if not os.path.exists(args.save_output_dir+'/'+ str(cam["camera"])):
                os.makedirs(args.save_output_dir+'/'+ str(cam["camera"]))


            
        '''
        cmd = 'ls ' + cam["tracking_dir"] + '/ | sort -n | tail -n 1'
        out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = out.communicate()[0]


        max_file = int(output[:-5])
        
        #print(max_file)


        #np.random.randint(2, size=args.num_objects)

        #input_config = [[0, [[60,2.5,200],[60,4,300], [60,7,50]]], [0, [[5,3,200]]], [0, [[40, 5, 100]]]]
        #input_config = get_config(cam["vehicles"])





        class_info[cam["camera"]] = get_class_info(cam["vehicles"], cam["tracking_dir"], max_file)


        data = get_track_points(cam["vehicles"], args.skip_files, cam["tracking_dir"],max_file)  
        '''
        
        episode_length = args.episode_len
        fps = args.fps
        
        desired_classes = [0,2]
        desired_tracks = [0,1,2,3]
        
        class_info[cam["camera"]] = get_class_info(cam["vehicles"], class_metadata[str(cam["camera"])], desired_classes)
             
        data = get_track_points(cam["vehicles"], args.skip_files, json_points[str(cam["camera"])], desired_tracks)
        
        track_points = process_points(data)
            

        synth_data[cam["camera"]], starting_time[cam["camera"]] = get_equally_spaced_points(cam["vehicles"], track_points, fps, args.episode_len,args.extra_vehicles)




        original_image[cam["camera"]] = cv2.imread(cam["display_image"])
        
                
        if args.create_video:
            pixel_height = original_image[cam["camera"]].shape[0]
            pixel_width = original_image[cam["camera"]].shape[1]
            video[cam["camera"]] = cv2.VideoWriter(args.create_video + str(cam["camera"]) + '.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 30, (pixel_width,pixel_height))


    #Print/save results
    

    
    for i in range(episode_length*fps):
        for cam in json_data["scenario"]:
            image = original_image[cam["camera"]].copy()
            
            to_delete = []
            create_file = False
            text_output = ''
            
            for s_idx,s in enumerate(starting_time[cam["camera"]]):
                if i >= s:
                    if i < s+synth_data[cam["camera"]][s_idx][2]:
                        display_point = [int(synth_data[cam["camera"]][s_idx][0][i-s]),int(synth_data[cam["camera"]][s_idx][1][i-s])]
                        display_point[0] = min(image.shape[1], display_point[0])
                        display_point[1] = min(image.shape[0], display_point[1])
                        
                    
                        #image = cv2.circle(image, display_point, radius=0, color=(0, 0, 255), thickness=10)

                        if args.display or args.create_video:
                            try:
                                display_point2 = [int(display_point[0]+class_info[cam["camera"]][synth_data[cam["camera"]][s_idx][3]]["box"][0]),int(display_point[1]+class_info[cam["camera"]][synth_data[cam["camera"]][s_idx][3]]["box"][1])]
                            except:
                                pdb.set_trace()
                            display_point2[0] = min(image.shape[1], display_point2[0])
                            display_point2[1] = min(image.shape[0], display_point2[1])
                            
                            crop_image = class_info[cam["camera"]][synth_data[cam["camera"]][s_idx][3]]["image"]

                            if "reverse" == synth_data[cam["camera"]][s_idx][6]:
                                crop_image = cv2.flip(crop_image,1)
                                
                            '''
                            if not i < synth_data[cam["camera"]][s_idx][7][0][1]:
                                synth_data[cam["camera"]][s_idx][7].pop(0)
                                
                            if synth_data[cam["camera"]][s_idx][7][0] and synth_data[cam["camera"]][s_idx][7][0][0] and i < synth_data[cam["camera"]][s_idx][7][0][1]:
                                crop_image = cv2.flip(crop_image,1)
                            '''
                            
                            alpha_channel = 3
                            crop_coords = np.where(crop_image[:,:,alpha_channel] > 0)

                            crop_coords1 = np.where(display_point[1] + crop_coords[0] < display_point2[1])
                            crop_coords2 = np.where(display_point[0] + crop_coords[1] < display_point2[0])
                            final_crop_coords = np.intersect1d(crop_coords1,crop_coords2)

                            #image[display_point[1]:display_point2[1],display_point[0]:display_point2[0]] = crop_image[0:display_point2[1]-display_point[1],0:display_point2[0]-display_point[0]]
                            image[display_point[1]+crop_coords[0][final_crop_coords],display_point[0]+crop_coords[1][final_crop_coords]] = crop_image[crop_coords[0][final_crop_coords],crop_coords[1][final_crop_coords],:alpha_channel]

                            #image = cv2.rectangle(image, display_point, display_point2, (0, 0, 255), 1)
                        
                        if args.save_output_dir:
                            text_output += "%d %f %f %f %f %d\n" % (synth_data[cam["camera"]][s_idx][3], *display_point,*display_point2, 1)
                            create_file = True
                    else:
                        to_delete.append(s_idx)
                        
            try:
                for d in to_delete:
                    del synth_data[cam["camera"]][d]
                    del starting_time[cam["camera"]][d]
            except:
                print("Error")
                    
            if args.save_output_dir and create_file:

                 f = open(args.save_output_dir+'/'+ str(cam["camera"]) + '/' + str(i) + '.txt', "w")
                 f.write(text_output)
                 f.close()
                    
            if args.display:
                cv2.imshow('image'+ str(cam["camera"]),image)
                cv2.waitKey(1)
                
            if args.create_video:
                video[cam["camera"]].write(image)

    for cam in json_data["scenario"]:
        if args.create_video:
            video[cam["camera"]].release()


