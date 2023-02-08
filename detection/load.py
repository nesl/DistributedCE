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

from motrackers import CentroidTracker

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
                results.append([results_tmp, state[t_key]['watchbox']['results'][results_tmp], t_key])
                
                
            
        
        
    return results,state
    
def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise RuntimeError("socket connection broken")
        data.extend(packet)
    return data    


yolo = Yolo_Exec(weights='./yolov5s.pt', imgsz=[800],conf_thres=0.5, device='1', save_conf=True) #'../../delivery-2022-12-01/t72detect_yv5n6.pt',imgsz=[2560],conf_thres=0.5)

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

#Two different tripwires defined as the location with respect to the image width for a given pixel column
#tripwire1 = 1300
#tripwire2 = 1750
#functions = ['cross_tripwire']



parser = argparse.ArgumentParser(description='Edge Processing Node')
parser.add_argument('--address', type=str, help='Address to connect to receive images')
parser.add_argument('--port', type=int, help='Port to connect to receive images')
parser.add_argument('--address_to_server', type=str, help='Address to connect to send atomic events')
parser.add_argument('--port_to_server', type=int, help='Port to connect to send atomic events')
parser.add_argument('--address_from_server', type=str, help='Address to connect to receive topic subscriptions')
parser.add_argument('--port_from_server', type=int, help='Port to connect to receive topic subscriptions')
parser.add_argument('--camera_id', type=int, help='Camera id')


args = parser.parse_args()


address = args.address
port = args.port

#function_metadata['cross_tripwire'] = [tripwire1,tripwire2]
'''
for f_idx in range(len(metadata)):
    #pdb.set_trace()
    
    inv_matrix_meta = metadata[f_idx]["CameraDatapoints"][1]["InverseViewProjectionMatrix"]
    pixel_width = metadata[f_idx]["CameraDatapoints"][1]["PixelWidth"]
    pixel_height = metadata[f_idx]["CameraDatapoints"][1]["PixelHeight"]
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

'''


client_socket = setup_socket()
client_socket.connect((address, port))

sock_from_server, sock_to_server = setup_zmq(args.address_from_server,args.address_to_server,args.port_from_server,args.port_to_server)

imgsz = 800*600*4

pixel_width = 800
pixel_height = 600
last_message_num = 0

first_pass = True
tracker = CentroidTracker(max_lost=5, tracker_output_format='mot_challenge')

decode_message = False

while True:    
    
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
    
    
    # Add the traffic camera label to it
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
    
    
    #pdb.set_trace()
    #print("received ", f_idx)
    #files[f_idx]
    res_lines = yolo.run(image)
    
    # Using cv2.putText() method
    image = cv2.putText(image, 'Traffic Camera ID: ' + str(args.camera_id), org, font, 
	           fontScale, color, thickness, cv2.LINE_AA)
    
    
    #pdb.set_trace()
    #res = open('exp5/labels/out-%04d.txt' %(f_idx+1))
    #res_lines = res.readlines()
 
    if not res_lines:
        cv2.imshow('image',image)
        cv2.waitKey(1)
 
 
    if res_lines:
        detection_bboxes = np.array([])
        detection_class_ids = np.array([])
        detection_confidences = np.array([])
        
        for line in res_lines:
            coordinates_line = line.split()
            box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),float(coordinates_line[4])), from_type="yolo", to_type="coco", image_size=(pixel_width,pixel_height))
            
            if detection_bboxes.size == 0:
                detection_bboxes = np.expand_dims(np.array(box_voc),axis=0)
            else:
                detection_bboxes = np.concatenate((detection_bboxes, np.expand_dims(np.array(box_voc),axis=0)),axis=0)
            #detection_bboxes = np.append(detection_bboxes, np.expand_dims(np.array(box_voc),axis=0),axis=0)
            detection_class_ids = np.append(detection_class_ids, int(coordinates_line[0]))
            detection_confidences = np.append(detection_confidences, float(coordinates_line[-1]))
            
        try:
            o_tracks = tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
            print(o_tracks)
        except:
            pdb.set_trace()
 
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
    
    for f in functions:
        #Apply functions according to query (right now only two tripwires are checked)
        res,state = eval(f+"(tracks,state,function_metadata['" + f +"'])")

        if res:
            new_res = {}
            new_res["camera_id"] = args.camera_id
            new_res["results"] = res
            new_res['time'] = f_idx
            
            if f == 'watchbox':
                print("Watchbox entered or not")
                
                sock_to_server.send_multipart([f.encode("utf-8"), pickle.dumps(new_res)])

                
                    
            if f == 'cross_tripwire': #If there is an event (change of state)
                print("Tripwires", r[0], "crossed by", r[1], "at", f_idx+1)
                #sock_to_server.send_multipart([topic.encode("utf-8"), pickle.dumps(events_by_topic[topic])])
                #pdb.set_trace()
                #screen_coordinates = ScreenToClipSpace((float(coordinates_line[1])*pixel_width, float(coordinates_line[2])*pixel_height, camera_height),pixel_width,pixel_height,far_clip_plane, near_clip_plane)
                #world_coordinates = np.matmul(inv_matrix,screen_coordinates)
                
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
