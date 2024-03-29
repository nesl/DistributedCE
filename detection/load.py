import json
import pdb
import numpy as np
from torchvision import ops
import torch
import cv2
from yolov5.detect_simple import Yolo_Exec
import glob
import os
import socket
import argparse
import pybboxes as pbx
from time import sleep

import pickle
import zmq
import time
import sys

sys.path.append('../network')
from data_format import Message, Data, Location

#from motrackers import CentroidTracker, IOUTracker, CentroidKF_Tracker, SORT
from trackers.tracker.byte_tracker import BYTETracker
from trackers.sort_tracker.sort import Sort
from trackers.motdt_tracker.motdt_tracker import OnlineTracker
from trackers.deepsort_tracker.deepsort import DeepSort

from sklearn.cluster import AgglomerativeClustering

def ScreenToClipSpace(screenPos,pixelWidth,pixelHeight,farClipPlane, nearClipPlane):

    w = screenPos[2]
    x = ((screenPos[0] / pixelWidth) * 2 - 1) * w
    y = ((screenPos[1] / pixelHeight) * 2 - 1) * w
    f = farClipPlane
    n = nearClipPlane
    z = (screenPos[2] * (f + n))/(f-n) - (2 * (f * n)) / (f - n)
    clipSpace = [x, y, z, w]
    
    return clipSpace


def setup_socket():


	
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5)

	
    return client_socket


def setup_zmq(ip_from_server,ip_to_server,port_from_server,port_to_server):

    ctx = zmq.Context()

    #server_ip = zmq_address  # "10.0.0.10" #"10.0.0.10" #"192.168.0.133"#"192.168.0.100" #"192.168.0.133" #"127.0.0.1"

    #port_from_server = "5561"
    #port_to_server = "5560"
    address_from_server = "tcp://%s:%s" % (ip_from_server, port_from_server)
    address_to_server = "tcp://%s:%s" % (ip_to_server, port_to_server)


    sock_from_server = ctx.socket(zmq.SUB)
    sock_from_server.connect(address_from_server)
    print("address_from_server:", address_from_server)
    sock_to_server = ctx.socket(zmq.PUB)
    sock_to_server.connect(address_to_server)
    sock_from_server.setsockopt(zmq.SUBSCRIBE, b'')
    print("address_to_server: ", address_to_server)
    
    return sock_from_server, sock_to_server




def state_init(state, track, functions, arguments):
    
    for f in functions:
        #state[track][f] = {'data': [], 'results': np.full((len(arguments[f]),), False)}
        
        if f not in state[track]:
            if f == 'convoy':
                state[track][f] = {'data': [], 'results': [[] for i in range(len(arguments[f]))]}
            else:
                state[track][f] = {'data': [], 'results': np.full((len(arguments[f]),), False)}
        else:
            new_args = np.full((len(arguments[f]),), False)
            state[track][f]['results'] = np.extend(state[track][f]['results'],new_args)
        
    return state
    
def state_add(state, functions, arguments):
    
    for s in state.keys():
        state = state_init(state, s, functions, arguments)
        
    return state

#Cross trip_wire function
def cross_tripwire(tracks,state, tripwires):
    results = []
    for t_key in tracks.keys():
        if state[t_key]['cross_tripwire']['data']:

            past_state = state[t_key]['cross_tripwire']['data'] - tripwires
            current_state = tracks[t_key][0][0] - tripwires
            results_tmp = np.where(np.sign(past_state)*np.sign(current_state) < 0)[0]

            
            if results_tmp.size > 0:
                #print(results_tmp, t_key)
                results.append([results_tmp, t_key])
            
        
        state[t_key]['cross_tripwire']['data'] = tracks[t_key][0][0]
        
    return results
    
#Watchbox function
def watchbox(tracks,state, watchboxes):
    results = []
    
    for t_key in tracks.keys():

            select = np.where(watchboxes[:,4] == tracks[t_key][2])[0]
            if select.size > 0:
                reference_point = [tracks[t_key][0][0]+(tracks[t_key][0][2] - tracks[t_key][0][0])/2, tracks[t_key][0][1] + (tracks[t_key][0][3] - tracks[t_key][0][1])/2]
                p1 = reference_point[0] - watchboxes[select,0] > 0
                p2 = reference_point[1] - watchboxes[select,1] > 0
                p3 = reference_point[0] - watchboxes[select,2] < 0
                p4 = reference_point[1] - watchboxes[select,3] < 0
                ptotal = p1 & p2 & p3 & p4

                
                try:
                    results_tmp = np.logical_xor(ptotal,state[t_key]['watchbox']['results'][select])
                except:
                    pdb.set_trace()
                    print("Error")
                #print(ptotal, state[t_key]['watchbox']['results'], results_tmp, t_key)
                results_tmp = np.nonzero(results_tmp)[0]
                results_tmp = select[results_tmp]
                state[t_key]['watchbox']['results'][select] = ptotal
                #state[t_key]['watchbox']['results'] = ptotal
                
                if results_tmp.size > 0:
                    #pdb.set_trace()
                    results.append([results_tmp.tolist(), state[t_key]['watchbox']['results'][results_tmp].tolist(), t_key])
                
    return results,state     
    
#Speed function    
def speed(tracks, state, speeds):
    results = []
    
    speed_threshold = 0.1
    
    for t_key in tracks.keys():
        select = np.where(speeds[:,1] == tracks[t_key][2])[0]
        reference_point = [tracks[t_key][0][0]+(tracks[t_key][0][2] - tracks[t_key][0][0])/2, tracks[t_key][0][1] + (tracks[t_key][0][3] - tracks[t_key][0][1])/2]
        
        if state[t_key]['speed']['data']:
            distance = np.linalg.norm(state[t_key]['speed']['data'] - reference_point)
            v = distance/(1/fps)
            ptotal = np.absolute(v - speeds[select,0]) > speed_threshold
            
            results_tmp = np.logical_xor(ptotal,state[t_key]['speed']['results'][select])

            results_tmp = np.nonzero(results_tmp)[0]
            results_tmp = select[results_tmp]
            state[t_key]['speed']['results'][select] = ptotal
            
            if results_tmp.size > 0:
                results.append([results_tmp, state[t_key]['speed']['results'][results_tmp], t_key])
        
        state[t_key]['speed']['data'] = reference_point


        
    return results,state
    
    
def convoy(tracks, state, groups):
    results = []
    

    if tracks:
        reference_points = np.array([[tracks[t_key][0][0]+(tracks[t_key][0][2] - tracks[t_key][0][0])/2, tracks[t_key][0][1] + (tracks[t_key][0][3] - tracks[t_key][0][1])/2, tracks[t_key][2], t_key] for t_key in tracks.keys()])
        min_number_vehicles = 3
        
        res_per_track_id = {}
        results_tmp = {}    
        for g_idx,g in enumerate(groups):
        
            rp_idxs = np.where(reference_points[:,2] == g[1])[0] 

            
            if len(reference_points) > min_number_vehicles and len(rp_idxs) > min_number_vehicles:
                

                clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=g[0], linkage='single').fit_predict(reference_points[rp_idxs,:2])

                labels,label_counts = np.unique(clustering, return_counts=True)
                #pdb.set_trace()
                count_res = np.where(label_counts > min_number_vehicles)[0]

                if count_res.size > 0:
                    
                    for possible_convoy in count_res:
                        track_idxs = np.where(clustering == labels[possible_convoy])[0]
                        convoy_elements = reference_points[rp_idxs[track_idxs],3]
                        if state[convoy_elements[0]]['convoy']['results'][g_idx]:
                            
                            existing_elements = set(state[convoy_elements[0]]['convoy']['results'][g_idx])
                            
                            if not existing_elements == set(convoy_elements): # and len(existing_elements.intersection(set(convoy_elements))) > 0: #This means the number of vehicles in the convoy has changed and the clustering corresponds to the previous one?
                            
                                merged_elements = existing_elements.union(set(convoy_elements))
                                
                                for t_key in merged_elements:
                                    #state[t_key]['convoy']['data'] = convoy_elements.tolist()
                                    
                                    
                                    #print(t_key, "changed to True modified")
                                    
                                    if t_key in convoy_elements:
                                        state[t_key]['convoy']['results'][g_idx] = convoy_elements.tolist()
                                    else:
                                        state[t_key]['convoy']['results'][g_idx] = []
                                        
                                    if t_key not in results_tmp:
                                        results_tmp[t_key] = []

                                    if t_key in convoy_elements:                                        
                                        results_tmp[t_key].append([g_idx, convoy_elements.tolist()])
                                    else:
                                        results_tmp[t_key].append([g_idx, []])
                                    #print(convoy_elements)
                        else:

                            for t_key in convoy_elements:
                                #state[t_key]['convoy']['data'] = convoy_elements.tolist()
                                
                                #print(t_key, "changed to True")
                                #print(convoy_elements)
                                state[t_key]['convoy']['results'][g_idx] = convoy_elements.tolist()
                                
                                if t_key not in results_tmp:
                                    results_tmp[t_key] = []
                                        
                                results_tmp[t_key].append([g_idx, convoy_elements.tolist()])
                                
                        
                    #print("Convoy detected", clustering)
                else:
                    for t_key in tracks.keys():
                        #state[t_key]['convoy']['data'] = []
                        if state[t_key]['convoy']['results'][g_idx]:
                            #print(t_key, "changed to False")
                            if t_key not in results_tmp:
                                results_tmp[t_key] = []
                            results_tmp[t_key].append([g_idx, []])

                        state[t_key]['convoy']['results'][g_idx] = []
                        
            else:
                for t_key in tracks.keys():
                    #state[t_key]['convoy']['data'] = []
                    if state[t_key]['convoy']['results'][g_idx]:
                        #print(t_key, "changed to False")
                        if t_key not in results_tmp:
                            results_tmp[t_key] = []
                        results_tmp[t_key].append([g_idx, []])

                    state[t_key]['convoy']['results'][g_idx] = []

        for result_key in results_tmp.keys():       

            #pdb.set_trace()
            results.append([[group[0] for group in results_tmp[result_key]], [group[1] for group in results_tmp[result_key]], result_key])
        
    
    return results, state
    
def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise RuntimeError("socket connection broken")
        data.extend(packet)
    return data    


def get_metadata_matrix(metadata, f_idx):

    inv_matrix_meta = metadata[f_idx]["CameraDatapoints"][1]["InverseViewProjectionMatrix"]
    #pixel_width = metadata[f_idx]["CameraDatapoints"][1]["PixelWidth"]
    #pixel_height = metadata[f_idx]["CameraDatapoints"][1]["PixelHeight"]
    near_clip_plane = metadata[f_idx]["CameraDatapoints"][1]["NearClipPlane"]
    far_clip_plane = metadata[f_idx]["CameraDatapoints"][1]["FarClipPlane"]
    camera_height = metadata[f_idx]["CameraDatapoints"][1]["CameraPosition"]["y"]
    inv_matrix = np.zeros((4,4))
    
    
    #We load the inverse view projection matrix
    inv_matrix[0,0] = inv_matrix_meta["m00"]
    inv_matrix[1,0] = inv_matrix_meta["m10"]
    inv_matrix[2,0] = inv_matrix_meta["m20"]
    inv_matrix[3,0] = inv_matrix_meta["m30"]
    inv_matrix[0,1] = inv_matrix_meta["m01"]
    inv_matrix[1,1] = inv_matrix_meta["m11"]
    inv_matrix[2,1] = inv_matrix_meta["m21"]
    inv_matrix[3,1] = inv_matrix_meta["m31"]
    inv_matrix[0,2] = inv_matrix_meta["m02"]
    inv_matrix[1,2] = inv_matrix_meta["m12"]
    inv_matrix[2,2] = inv_matrix_meta["m22"]
    inv_matrix[3,2] = inv_matrix_meta["m32"]
    inv_matrix[0,3] = inv_matrix_meta["m03"]
    inv_matrix[1,3] = inv_matrix_meta["m13"]
    inv_matrix[2,3] = inv_matrix_meta["m23"]
    inv_matrix[3,3] = inv_matrix_meta["m33"]
    
    return inv_matrix, camera_height, far_clip_plane, near_clip_plane
    
    
def get_ground_truths(ground_truth_file, previous_ground_truth, f_idx): #If bbox is in first frame, problems

    ground_truth_list = []
    


    if previous_ground_truth:
        if int(previous_ground_truth[2]) == f_idx:
            ground_truth_list.append(previous_ground_truth)

            while True:
                
                next_ground_truth_tmp = ground_truth_file.readline()
                
                if len(next_ground_truth_tmp) == 0:
                    break
                    
                next_ground_truth = next_ground_truth_tmp.strip().split(',')
            
                if int(next_ground_truth[2]) == f_idx:
                    ground_truth_list.append(next_ground_truth)
                else:
                    break
        else:
            next_ground_truth = previous_ground_truth
        
    else:
        next_ground_truth = ground_truth_file.readline().strip().split(',')
                
    return ground_truth_list, next_ground_truth
    
def rewind_ground_truths(ground_truth_file, camera_num):

    while True:
        next_ground_truth_tmp = ground_truth_file.readline()
            
        next_ground_truth = next_ground_truth_tmp.strip().split(',')
        
        if str(camera_num) in next_ground_truth[3]:
            break
            
    return next_ground_truth
    

#source = '../../Delivery-2022-12-12/video1/'
#files = sorted(glob.glob(os.path.join(source, '*.*')))


#Metadata with camera information
#f = open('../../Delivery-2022-12-12/metadata.json')

#metadata = json.load(f)

tracks = {}
state = {}

track_id = 0

function_metadata = {}
functions = []

fps = 10

#Two different tripwires defined as the location with respect to the image width for a given pixel column
#tripwire1 = 1300
#tripwire2 = 1750
#functions = ['convoy']
#function_metadata['convoy'] = [[80,0],[200,1]]
#functions = ['watchbox']
#function_metadata['watchbox'] = np.array([[213,274,772,772,0], [816,366,1200,725,0], [1294,290,1881,765,0]]) #np.array([[20,147,340,520,1],[20,147,340,520,0]]) #np.array([[20,130,290,560,0]]) #np.array([[20,147,340,520,1]]) #np.array([[213,274,772,772,0], [816,366,1200,725,0], [1294,290,1881,765,0]])
write_to_file_res = []

#new_function = {'watchbox':function_metadata['watchbox']}
#state = state_add(state,['watchbox'],new_function)

parser = argparse.ArgumentParser(description='Edge Processing Node')
parser.add_argument('--address', type=str, help='Address to connect to receive images')
parser.add_argument('--port', type=int, help='Port to connect to receive images')
parser.add_argument('--address_to_server', type=str, help='Address to connect to send atomic events')
parser.add_argument('--port_to_server', type=int, help='Port to connect to send atomic events')
parser.add_argument('--address_from_server', type=str, help='Address to connect to receive topic subscriptions')
parser.add_argument('--port_from_server', type=int, help='Port to connect to receive topic subscriptions')
parser.add_argument('--camera_id', type=int, help='Camera id')
parser.add_argument('--take', default=0, type=int, help='Camera id')
parser.add_argument('--track_alg', type=str, default='Byte', help='Track algorithm: Byte, Sort, DeepSort or MOTDT')
parser.add_argument('--video-file', type=str, default='', help='Open video file instead of connecting to server')
parser.add_argument('--image-file', type=str, default='', help='Open image file instead of connecting to server')
parser.add_argument('--yolo-weights', nargs='+', type=str, default='./yolov5s.pt', help="YOLO weights file")
parser.add_argument('--device', type=str, default='0', help="Device where to run YOLO")
parser.add_argument('--metadata', type=str, help='Camera metadata file')
parser.add_argument('--save-tracking-dir', type=str, default='', help='Save tracking results to the directory specified')
parser.add_argument('--save-raw-yolo-dir', type=str, default='', help='Save raw yolo results to the directory specified')
parser.add_argument('--create-video', type=str, default='', help='Create video. Specify file name.')
parser.add_argument('--yolo-synth-output', type=str, default='', help='Provide YOLO files and bypass the model. Specify the directory name.')
parser.add_argument('--display', action='store_true', help='Display video')
parser.add_argument('--ground-truth', type=str, default='', help='Provide ground truth csv file')
parser.add_argument('--start', type=int, default=0, help='Starting frame if using video')
parser.add_argument('--end', type=int, default=0, help='Ending frame if using video')
parser.add_argument('--conf-thres', type=float, default=0.1, help='YOLO confidence threshold')


args = parser.parse_args()




address = args.address
port = args.port


f_idx = -1

if args.video_file:
    cap = cv2.VideoCapture(args.video_file)
    pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if args.start:
        f_idx = args.start-1
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    
    if args.metadata:
        f_metadata = open(args.metadata)

        metadata = json.load(f_metadata)
        
elif args.image_file:
    image = cv2.imread(args.image_file)
    pixel_height = image.shape[0]
    pixel_width = image.shape[1]
else:
    client_socket = setup_socket()
    client_socket.connect((address, port))
    sock_from_server, sock_to_server = setup_zmq(args.address_from_server,args.address_to_server,args.port_from_server,args.port_to_server)
    pixel_width = 800
    pixel_height = 600
    imgsz = pixel_width*pixel_height*4

print("Image width: ", pixel_width, " image height: ", pixel_height, " fps: ", fps)

if not args.yolo_synth_output:
    yolo = Yolo_Exec(weights=args.yolo_weights, imgsz=[pixel_width],conf_thres=args.conf_thres, device=args.device, save_conf=True) #'../../delivery-2022-12-01/t72detect_yv5n6.pt',imgsz=[2560],conf_thres=0.5)
    #yolo = Yolo_Exec(weights=args.yolo_weights, imgsz=[pixel_width],conf_thres=0.1, device=args.device, save_conf=True)

last_message_num = 0

first_pass = True
#tracker = SORT(max_lost=20, iou_threshold=0.0001, tracker_output_format='mot_challenge')
#tracker = CentroidKF_Tracker(max_lost=20)

decode_message = False

old_tracks = []

class ByteTrackArgs:
    def __init__(self, track_thresh, match_thresh, track_buffer, low_track_thres):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.mot20 = False 
        self.track_buffer = track_buffer
        self.low_track_thres = low_track_thres



#Choose tracking algorithm
track_alg = args.track_alg

if track_alg == 'Byte':
    new_args = ByteTrackArgs(0.5,0.8, 100, 0.1)#20)
    tracker = BYTETracker(new_args, frame_rate=fps)
elif track_alg == 'Sort':
    tracker = Sort(new_args.track_thresh)
elif track_alg == 'MOTDT':
    tracker = OnlineTracker('trackers/pretrained/googlenet_part8_all_xavier_ckpt_56.h5', use_tracking=False)
elif track_alg == 'DeepSort':
    tracker = DeepSort('trackers/pretrained/ckpt.t7', max_age=100)
    
#Create directory for saving tracking results
if args.save_tracking_dir:
    if not os.path.exists(args.save_tracking_dir):
        os.makedirs(args.save_tracking_dir)
        
#Create directory for saving raw yolo results
if args.save_raw_yolo_dir:
    if not os.path.exists(args.save_raw_yolo_dir):
        os.makedirs(args.save_raw_yolo_dir)

#Create video file
if args.create_video:
    video = cv2.VideoWriter(args.create_video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (pixel_width,pixel_height))

    


#Ttry to match ground truth annotations with detections made with yolo here
if args.ground_truth:
    ground_truth_file = open("take_201/neuroplexLog_201_objectvisibility.csv")
    ground_truth_file.readline()

    next_ground_truth = rewind_ground_truths(ground_truth_file, args.camera_id)

    ground_truth_correspondence = {}

while True:    
    
    #Non-blocking check for commands from server (primitive events to generate)
    try:
        topic, msg = sock_from_server.recv_multipart(flags=zmq.NOBLOCK)
        print("Receive message")
        print(
            '   Topic: {}, msg:{}'.format(
                topic.decode('utf-8'), pickle.loads(msg)
            )
        )
        decode_message = True
        
    except:
        pass
        
    if decode_message:
        decode_message = False
        message = pickle.loads(msg)
        print("Message", message)
        if message.message_number > last_message_num: # and message.area == args.camera_id: #We process request only if it corresponds to the area the camera is in
            atomic_event = message.topics
            print("Atomic event", atomic_event)
            if atomic_event not in functions:
                functions.append(atomic_event)
            print("jere")
            if atomic_event in function_metadata:
                function_metadata[atomic_event] = np.extend(function_metadata[atomic_event], np.array(message.arguments))
            else:
                function_metadata[atomic_event] = np.array(message.arguments)
            print("here2", message.arguments)
            new_function = {atomic_event:np.array(message.arguments)}
            try:
                state = state_add(state,[atomic_event],new_function)
                print('state', state)
            except:
                pdb.set_trace()
            last_message_num = message.message_number

            
        
    
    #Open from video file or receive frames from network
    if args.video_file:
        
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Stream closed")
            break
        
        if args.end and f_idx == args.end:
            break
        ret, image = cap.read()
        if not ret:
            print("End of video stream")
            cap.release()
            break
        f_idx += 1
    elif args.image_file:
        pass
    else:
        try:
            f_idx = int.from_bytes(recvall(client_socket, 2),'big')

            image =recvall(client_socket, imgsz)
        except:
            print('Timeout')
            #server_connection, addr = server_socket.accept()
            client_socket = setup_socket()
            connected = False
            while not connected:  
                # attempt to reconnect, otherwise sleep for 2 seconds  
                try:  
                    client_socket.connect((address, port))
                    connected = True  
                    print( "re-connection successful" )  
                except socket.error:  
                    sleep( 1 ) 
            #server_connection.settimeout(5)
            continue	

        #imageidx = int.from_bytes(server_connection.recv(2),"big")
        image_np = np.frombuffer(image, dtype=np.dtype("uint8"))#.reshape(2560,1440)
        image = cv2.cvtColor( image_np.reshape(600,800,4), cv2.COLOR_BGRA2BGR )
    
    #Required for some tracking algorithms that rely on reid
    if track_alg == 'MOTDT' or track_alg == 'DeepSort':
        image2 = image.copy()
    
    
    
    
    
    #pdb.set_trace()
    #print("received ", f_idx)
    #files[f_idx]
    #time_past = time.time()
    
    res_lines = []
    
    #You can provide a directory with files that correspond to frames and detections made on those frames, and therefore load those results instead of relying on yolo
    if args.yolo_synth_output:
        #filename = args.yolo_synth_output + '/' + str(args.camera_id) + '/' + str(f_idx) + '.txt'
        filename = args.yolo_synth_output + '/' + str(args.take) + "_" + str(args.camera_id) + "_" + str(f_idx) + '.txt'
        #filename = args.yolo_synth_output + '/' + str(f_idx) + '.txt'
        
        if os.path.exists(filename):
            yolo_out = open(filename).readlines()
            res_lines = list(map(lambda s: s.strip(), yolo_out))

        
    else:
        #start_time = time.time()
        #if f_idx == 2077:
        #    pdb.set_trace()
        res_lines = yolo.run(image) #Run yolo
        #print(time.time()-start_time)
    #print("Yolo exec time:", time.time()-time_past)
    
    
    # Add the traffic camera label to images
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # COLOR in BGR
    color = (0, 255, 0)
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    image = cv2.putText(image, 'Traffic Camera ID: ' + str(args.camera_id) + ' Frame: ' + str(f_idx), org, font, 
	           fontScale, color, thickness, cv2.LINE_AA)
    
    
    #pdb.set_trace()
    #res = open('exp5/labels/out-%04d.txt' %(f_idx+1))
    #res_lines = res.readlines()
 
 
    if args.ground_truth:
        if str(args.camera_id) not in next_ground_truth[3]:
            ground_truths = []
            break #End when we don't have more ground truths to evaluate
        else:
            ground_truths, next_ground_truth = get_ground_truths(ground_truth_file, next_ground_truth, f_idx)
            
        if ground_truths:
            iou_score = 0
            max_index = -1
            g_bboxes = []
            g_obj_ids = []
            for g_idx,g_bbox in enumerate(ground_truths):
                g_bboxes.append([float(g_bbox[11].replace('(','')),pixel_height-float(g_bbox[12].replace(')','')),float(g_bbox[9].replace('(','')),pixel_height-float(g_bbox[10].replace(')',''))])
                g_obj_ids.append(g_bbox[5])
                
                
                if g_bbox[5] not in ground_truth_correspondence:
                    ground_truth_correspondence[g_bbox[5]] = {"results":[],"frames":[]}
                ground_truth_correspondence[g_bbox[5]]["results"].append([-1]*3)
                ground_truth_correspondence[g_bbox[5]]["frames"].append(f_idx)
                    
                    
            g_bboxes = torch.as_tensor(g_bboxes)
            
            
            
            #ground_truth_correspondence = [[-1]*3]*len(ground_truths)
        
    #print(tracks, f_idx)
    time_past = time.time()
    if res_lines:
        #print(res_lines, f_idx)
        detection_bboxes = np.array([])
        detection_class_ids = np.array([])
        detection_confidences = np.array([])
        detection_extra = np.array([])
        
        text_output = ''
        
        #Every yolo detection is processed here, each line is a detection
        for line in res_lines:
        
            if args.save_raw_yolo_dir: #To save the raw yolo detection into files

                splitted_line = line.split()
                raw_line = ' '.join(splitted_line[:5]) + ' ' + splitted_line[6] + ' ' + str(max(map(float,splitted_line[7:]))) + '\n'
                text_output += raw_line
        
            coordinates_line = line.split()

            #if int(coordinates_line[0]) > 2: #Only pedestrians
            #    continue
            
            #if args.yolo_synth_output:

            #    box_voc = (int(float(coordinates_line[1])),int(float(coordinates_line[2])),int(float(coordinates_line[3])),int(float(coordinates_line[4])))
            #else:
            
            try: #Convert yolo bbox format to voc
                box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),float(coordinates_line[4])), from_type="yolo", to_type="voc", image_size=(pixel_width,pixel_height))
            except:
                print("Error dimensions")
                continue
               
            
            cv2.rectangle(image, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)
            
            
            if args.yolo_synth_output:
                continue
            
            if detection_bboxes.size == 0:
                detection_bboxes = np.expand_dims(np.array(box_voc),axis=0)
            else:
                detection_bboxes = np.concatenate((detection_bboxes, np.expand_dims(np.array(box_voc),axis=0)),axis=0)
            #detection_bboxes = np.append(detection_bboxes, np.expand_dims(np.array(box_voc),axis=0),axis=0)
            detection_class_ids = np.append(detection_class_ids, int(coordinates_line[0]))
            detection_confidences = np.append(detection_confidences, float(coordinates_line[5]))
            
            #Extra data is the distribution of class probabilities
            extra_data = np.array([float(cl) for cl in coordinates_line[6:]])
            if detection_extra.size == 0:
                detection_extra = np.expand_dims(extra_data,axis=0)
            else:
                detection_extra = np.concatenate((detection_extra,np.expand_dims(extra_data,axis=0)),axis=0 )
                
            if args.ground_truth:
                if ground_truths:

                    box_voc_tensor = torch.as_tensor([box_voc])
      
                    iou = ops.box_iou(g_bboxes, box_voc_tensor)
                    iou_max = torch.max(iou,dim=0)
                    iou = iou_max[0][0]
                    box_idx = iou_max[1][0]
                    
                    #if f_idx == 4534:
                    #    pdb.set_trace()

                    #pdb.set_trace()
                    if iou > 0.2:
                    
                        #if g_obj_ids[box_idx] == ' -900348' and f_idx == 2058:
                        #    pdb.set_trace()

                        

                        if int(coordinates_line[0]) == int(ground_truths[box_idx][6]):
                            ground_truth_correspondence[g_obj_ids[box_idx]]["results"][-1] = [1, -1, -1]
                        else:

                            if ground_truth_correspondence[g_obj_ids[box_idx]]["results"][-1][0] != 1:
                                ground_truth_correspondence[g_obj_ids[box_idx]]["results"][-1][0] = 0
                                ground_truth_correspondence[g_obj_ids[box_idx]]["results"][-1][1] = int(coordinates_line[0])
                                ground_truth_correspondence[g_obj_ids[box_idx]]["results"][-1][2] = int(np.where(np.flip(np.argsort(extra_data[1:])) == int(ground_truths[box_idx][6]))[0][0])
                            
                                if ground_truth_correspondence[g_obj_ids[box_idx]]["results"][-1][2] == 0:
                                    pdb.set_trace()
                                
                            
        if args.save_raw_yolo_dir:
             f = open(args.save_raw_yolo_dir + '/' + str(f_idx) + '.txt', "w")
             f.write(text_output)
             f.close()    
        #print(detection_bboxes)
        #o_tracks = tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
        #pdb.set_trace()
        
        
        text_output = ''
        
        if detection_bboxes.size > 0: #If we got bounding boxes for the current frame let's associate them with tracks
        
            #Choose the tracking algorithm           
            if track_alg == 'MOTDT' or track_alg == 'DeepSort': #Only use DeepSort
                online_targets,extra_outputs = tracker.update(np.column_stack((detection_bboxes,detection_confidences, detection_class_ids, detection_extra)), [pixel_height, pixel_width], (pixel_height,pixel_width), image2)
            else: #Only use ByteTracker
                bbox_stack = np.column_stack((detection_bboxes,detection_confidences))
                #print(bbox_stack, detection_class_ids)

                online_targets,_,_ = tracker.update(bbox_stack, [pixel_height, pixel_width], (pixel_height,pixel_width), detection_class_ids, detection_extra) #check order of data error
                
            
            
            new_tracks = []
            #pdb.set_trace()
            for t_idx,t in enumerate(online_targets): #Get the resultant tracks
                
                if track_alg == 'Sort' or track_alg == 'DeepSort':
                    track_id = int(t[4])
                    bbox = t[:4]
                    class_history = extra_outputs[t_idx][0]
                    class_extra = extra_outputs[t_idx][1]
                    #pdb.set_trace()
                else:
                    track_id = t.track_id
                    bbox = t.tlbr
                    class_history = t.detected_class #For Bytetracker, haven't tested with other ones
                    detection_extra = t.detected_extra
                
     
                
                
                new_tracks.append(track_id)
                

                
                class_detected = np.bincount(class_history).argmax()
                
                
                tracks[track_id] = (bbox,f_idx,class_detected,class_history,detection_extra)
                if track_id not in state:
                    state[track_id] = {}
                    state = state_init(state,track_id,functions,function_metadata)
                    
                #Put label outside bounding box
                fontScale = 0.5
                color = (255, 153, 255)
                image = cv2.putText(image, str(track_id), (int(bbox[0]),int(bbox[1])), font, 
                               fontScale, color, thickness, cv2.LINE_AA)
                               
                if args.save_tracking_dir:
                    text_output += "%f,%f,%f,%f,%d,%d\n" % (*bbox,track_id, class_history[-1]) #class_detected)
                
            
            if set(new_tracks) != set(old_tracks):
                print("New tracks:",  new_tracks, online_targets)
                for tt in list(set(old_tracks) - set(new_tracks)):
                    del tracks[tt]

                
            old_tracks = new_tracks
            
            if args.save_tracking_dir:
                 f = open(args.save_tracking_dir + '/' + str(f_idx) + '.txt', "w")
                 f.write(text_output)
                 f.close()
                 
            if args.create_video:
                video.write(image)
            
                
            """
            new_tracks = []
            for ot in o_tracks:    
                new_tracks.append(ot[1])
            if set(new_tracks) != set(old_tracks):
                print("New tracks:",  new_tracks, o_tracks)
                
            old_tracks = new_tracks
            #print(o_tracks)
            """
        '''
        
        
    
 
    #We read the detection file and iterate over the bounding box lines
    for line in res_lines:
        max_overlap = [0,0]

        coordinates_line = line.split()
        object_class = int(coordinates_line[0])
        
        box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),float(coordinates_line[4])), from_type="yolo", to_type="voc", image_size=(pixel_width,pixel_height))
        
        #success, bboxes = multi_tracker.update(image)
        #if not first_pass and success:
        #    cv2.rectangle(image, (bboxes[0], bboxes[1]), (bboxes[0]+bboxes[2], bboxes[1]+bboxes[3]), (0, 0, 255), 1)
        cv2.rectangle(image, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)
        cv2.imshow('image',image)
        cv2.waitKey(1)


        new_box = np.array([box_voc[0],box_voc[1],box_voc[2],box_voc[3]],dtype=np.float32)
        #print(box_voc, new_box)

        #if first_pass:        
        #    multi_tracker.add(cv2.TrackerCSRT_create(), image, np.array([box_voc[0],box_voc[1],box_voc[2]-box_voc[0],box_voc[3]-box_voc[1]],dtype=np.float32))
        #    first_pass = False
        
        
        
        #We try tracking the objects using IoU
        if not tracks: #Create track for a new object
            tracks[track_id] = (new_box,f_idx, object_class)
            state[track_id] = {}
            state = state_init(state,track_id,functions,function_metadata)
            print("New track 1", track_id, state[track_id], object_class)
            track_id += 1
        else:
            for t_key in tracks.keys():
                #pdb.set_trace()
                overlap = ops.box_iou(torch.from_numpy(tracks[t_key][0]).unsqueeze(0), torch.from_numpy(new_box).unsqueeze(0))[0][0]
                if overlap > max_overlap[1]:
                    max_overlap[1] = overlap
                    max_overlap[0] = t_key
            if max_overlap[1] <= 0: #If htere is no that much overlap, create new track
                tracks[track_id] = (new_box,f_idx, object_class)
                state[track_id] = {}
                state = state_init(state,track_id,functions,function_metadata)
                print("New track 2", track_id, state[track_id], max_overlap[1], object_class)
                track_id += 1
                
            else:
                tracks[max_overlap[0]] = (new_box,f_idx, tracks[max_overlap[0]][2]) #keep object class or assign new one?
                
    #Delete tracks that are not relevant anymore
    to_delete = []        
    for t_key in tracks.keys():
        if f_idx - tracks[t_key][1] > 5:
            to_delete.append(t_key)


    for t_key in to_delete:
        del tracks[t_key]
        
        del state[t_key]
        
    #print(tracks)
    '''
    

    
    if args.display:
        cv2.imshow('image',image)
        
        if args.image_file:
            cv2.waitKey(10000)
            break
        
        cv2.waitKey(1)

    for f in functions: #This is to evaluate different functions of interest over the tracks
        #Apply functions according to query (right now only two tripwires are checked)
        res,state = eval(f+"(tracks,state,function_metadata['" + f +"'])")

        if res:
            new_res = {}
            new_res["camera_id"] = args.camera_id
            new_res["results"] = res
            new_res['time'] = f_idx
            
            if f == 'watchbox':
                print("Watchbox entered or not")
                print(new_res)
                write_to_file_res.append(new_res)
                
                #sock_to_server.send_multipart([f.encode("utf-8"), pickle.dumps(new_res)])

                
                    
            if f == 'cross_tripwire': #If there is an event (change of state)
                print("Tripwires", r[0], "crossed by", r[1], "at", f_idx+1)
                #sock_to_server.send_multipart([topic.encode("utf-8"), pickle.dumps(events_by_topic[topic])])
                #pdb.set_trace()
                #screen_coordinates = ScreenToClipSpace((float(coordinates_line[1])*pixel_width, float(coordinates_line[2])*pixel_height, camera_height),pixel_width,pixel_height,far_clip_plane, near_clip_plane)
                #world_coordinates = np.matmul(inv_matrix,screen_coordinates)
                
                
            if f == 'convoy':
                m, s = divmod(f_idx/fps, 60)
                h, m = divmod(m, 60)
                print("Convoy", new_res, m,s)
                
    #image_res = cv2.imread('exp5/out-%04d.jpg' %(f_idx+1))

    #cv2.imshow("result", image_res)
    #cv2.waitKey(1)

    """
    coordinates_line = line.split()
    screen_coordinates = np.array([float(coordinates_line[1])*pixel_width,1,float(coordinates_line[2])*pixel_height,1])
    world_coordinates = np.matmul(screen_coordinates,inv_matrix)
    world_coordinates /= world_coordinates[-1]
    print(world_coordinates)
    pdb.set_trace()
    """
    
    


#ce_json = open("complex_results.json","w")
#json.dump(write_to_file_res,ce_json)

if args.ground_truth:

    error_results = {}

    for g_key in ground_truth_correspondence.keys():
        results = np.array(ground_truth_correspondence[g_key]["results"])
        num_results = results.shape[0]
        accuracy_object = len(np.where(results[:,0] == 1)[0])/num_results
        
        error_results[g_key] = {}
        
        print("accuracy for ", g_key, " ", accuracy_object)
        
        error_results[g_key]["accuracy"] = float(accuracy_object)
        
        values,counts = np.unique(results[:,1], return_counts=True)
        
        c_indices = np.where(values > -1)[0]
        
        total_missc = np.sum(counts[c_indices])

        missclass_dict = {int(values[c_idx]):float(counts[c_idx]/total_missc) for c_idx in c_indices}
        
        print("missclassification for ", g_key, " ", missclass_dict)
        
        error_results[g_key]["missclass"] = missclass_dict
        
        values,counts = np.unique(results[:,2], return_counts=True)
        c_indices = np.where(values > -1)[0]
        total_misrank = np.sum(counts[c_indices])
        
        missrank_dict = {int(values[c_idx]+1):float(counts[c_idx]/total_misrank) for c_idx in c_indices}
        
        print("missrank for ", g_key, " ", missrank_dict)
        
        error_results[g_key]["missrank"] = missrank_dict
        
        print(ground_truth_correspondence[g_key])
        
        error_results[g_key]["data"] = ground_truth_correspondence[g_key]


    with open(args.video_file[:-4] + "_error.json", "w") as outfile:
        try:
            json.dump(error_results, outfile)
        except:
            pdb.set_trace()
    
    
if args.create_video:
    video.release()
