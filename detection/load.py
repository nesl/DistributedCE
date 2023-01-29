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

def setup_connections_and_handling(address, port):

	# Insert our networking stuff
	print("Setting up Server...")
	server_connections = []
	
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server_socket.bind((address, port))
	server_socket.listen()

	server_connection, addr = server_socket.accept()

	# server_socket.connect(("127.0.0.1", 55000))
	print("Server set up...")
	
	return server_connection


def state_init(state, functions):
    
    for s in state.keys():
        for f in functions:
            state[s][f] = {'data': [], 'results': False}


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
    
def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data    


yolo = Yolo_Exec(weights='./yolov5s.pt', imgsz=[800],conf_thres=0.5) #'../../delivery-2022-12-01/t72detect_yv5n6.pt',imgsz=[2560],conf_thres=0.5)

source = '../../Delivery-2022-12-12/video1/'
files = sorted(glob.glob(os.path.join(source, '*.*')))


#Metadata with camera information
#f = open('../../Delivery-2022-12-12/metadata.json')

#metadata = json.load(f)

tracks = {}
state = {}

track_id = 0

function_metadata = {}


#Two different tripwires defined as the location with respect to the image width for a given pixel column
tripwire1 = 1300
tripwire2 = 1750
functions = ['cross_tripwire']



parser = argparse.ArgumentParser(description='Forwarder.')
parser.add_argument('--address', type=str, help='Address to bind')
parser.add_argument('--port', type=int, help='Port to bind')


args = parser.parse_args()


address = args.address
port = args.port

function_metadata['cross_tripwire'] = [tripwire1,tripwire2]
'''
for f_idx in range(len(metadata)):
    #pdb.set_trace()
    
    inv_matrix_meta = metadata[f_idx]["CameraDatapoints"][1]["InverseViewProjectionMatrix"]
    pixel_width = metadata[f_idx]["CameraDatapoints"][1]["PixelWidth"]
    pixel_height = metadata[f_idx]["CameraDatapoints"][1]["PixelHeight"]
    near_clip_plane = metadata[f_idx]["CameraDatapoints"][1]["NearClipPlane"]
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
    """
    inv_matrix[0,0] = inv_matrix_meta["m00"]
    inv_matrix[0,1] = inv_matrix_meta["m10"]
    inv_matrix[0,2] = inv_matrix_meta["m20"]
    inv_matrix[0,3] = inv_matrix_meta["m30"]
    inv_matrix[1,0] = inv_matrix_meta["m01"]
    inv_matrix[1,1] = inv_matrix_meta["m11"]
    inv_matrix[1,2] = inv_matrix_meta["m21"]
    inv_matrix[1,3] = inv_matrix_meta["m31"]
    inv_matrix[2,0] = inv_matrix_meta["m02"]
    inv_matrix[2,1] = inv_matrix_meta["m12"]
    inv_matrix[2,2] = inv_matrix_meta["m22"]
    inv_matrix[2,3] = inv_matrix_meta["m32"]
    inv_matrix[3,0] = inv_matrix_meta["m03"]
    inv_matrix[3,1] = inv_matrix_meta["m13"]
    inv_matrix[3,2] = inv_matrix_meta["m23"]
    inv_matrix[3,3] = inv_matrix_meta["m33"]
    """
'''


server_connection = setup_connections_and_handling(address,port)

imgsz = 800*600*4

pixel_width = 800
pixel_height = 600

while True:    
    
    f_idx = int.from_bytes(recvall(server_connection, 2),'big')

    image =recvall(server_connection, imgsz)

    #imageidx = int.from_bytes(server_connection.recv(2),"big")
    image_np = np.frombuffer(image, dtype=np.dtype("uint8"))#.reshape(2560,1440)
    image = cv2.cvtColor( image_np.reshape(600,800,4), cv2.COLOR_BGRA2BGR )
    cv2.imshow('image',image)
    cv2.waitKey(1)
    #pdb.set_trace()
    print("received ", f_idx)
    #files[f_idx]
    res_lines = yolo.run(image)
    #pdb.set_trace()
    #res = open('exp5/labels/out-%04d.txt' %(f_idx+1))
    #res_lines = res.readlines()
 
    #We read the detection file and iterate over the bounding box lines
    for line in res_lines:
        max_overlap = [0,0]

        coordinates_line = line.split()
        
        
        
        
        x1 = float(coordinates_line[1])*pixel_width
        y1 = float(coordinates_line[2])*pixel_height
        
        x2 = x1 + float(coordinates_line[3])*pixel_width
        y2 = y1 + float(coordinates_line[4])*pixel_height

        new_box = np.array([x1,y1,x2,y2],dtype=np.float32)
        
        
        #We try tracking the objects using IoU
        if not f_idx or not tracks: #Create track for a new object
            tracks[track_id] = (new_box,f_idx)
            state[track_id] = {}
            state_init(state,functions)
            
            track_id += 1
        else:
            for t_key in tracks.keys():
                #pdb.set_trace()
                overlap = ops.box_iou(torch.from_numpy(tracks[t_key][0]).unsqueeze(0), torch.from_numpy(new_box).unsqueeze(0))[0][0]
                if overlap > max_overlap[1]:
                    max_overlap[1] = overlap
                    max_overlap[0] = t_key
            if max_overlap[1] < 0.2: #If htere is no that much overlap, create new track
                tracks[track_id] = (new_box,f_idx)
                state[track_id] = {}
                state_init(state,functions)
                track_id += 1
            else:
                tracks[max_overlap[0]] = (new_box,f_idx)
                
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
        res = eval(f+"(tracks,state,function_metadata['" + f +"'])")

    
        if res:
            for r in res:
                if f == 'cross_tripwire': #If there is an event (change of state)
                    print("Tripwires", r[0], "crossed by", r[1], "at", f_idx+1)
                    #pdb.set_trace()
                    #Range of values where tank should be x = [2697,2938], z = [-2127,-1698]
                    #Not sure about this
                    '''
                    screen_coordinates = np.array([float(coordinates_line[1])*pixel_width,float(coordinates_line[2])*pixel_height,near_clip_plane,1]) #According to https://gamedevbeginner.com/how-to-convert-the-mouse-position-to-world-space-in-unity-2d-3d/ and https://docs.unity3d.com/ScriptReference/Camera.ScreenToWorldPoint.html the z value should be the near_clip_plane

                    #screen_coordinates = np.array([float(coordinates_line[2])*pixel_height,float(coordinates_line[1])*pixel_width,near_clip_plane,1])
                    world_coordinates = np.matmul(inv_matrix,screen_coordinates) #np.matmul(inv_matrix,screen_coordinates) #np.matmul(screen_coordinates,inv_matrix)
                    normalized_world_coordinates = world_coordinates/world_coordinates[-1]
                    print(normalized_world_coordinates)
                    '''
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
