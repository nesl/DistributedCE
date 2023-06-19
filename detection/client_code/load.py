# import json
# import pdb
import numpy as np
# from torchvision import ops
# import torch
import cv2
# from yolov5.modified_detect_simple import Yolo_Exec
import glob
import os
import socket
from socket import SHUT_RDWR
import argparse
import pybboxes as pbx
from time import sleep

import pickle
# import zmq
import time
import sys
import csv
from tqdm import tqdm
import torch
import torchvision

from yolov5_core import Yolov5 as Yolov5Trt
from vidsz.opencv import Reader, Writer


# Get our ground truth data
# from gt_files.gt_align import gt_source_drone

import traceback

import warnings
warnings.filterwarnings("ignore")

# from torch.multiprocessing import Process, set_start_method
# set_start_method('spawn', force=True)

# sys.path.append('../network')
# from data_format import Message, Data, Location

#from motrackers import CentroidTracker, IOUTracker, CentroidKF_Tracker, SORT
from byte_tracker.byte_tracker import BYTETracker
# from trackers.sort_tracker.sort import Sort
# from trackers.motdt_tracker.motdt_tracker import OnlineTracker
# from trackers.deepsort_tracker.deepsort import DeepSort

# from sklearn.cluster import AgglomerativeClustering

def ScreenToClipSpace(screenPos,pixelWidth,pixelHeight,farClipPlane, nearClipPlane):

    w = screenPos[2]
    x = ((screenPos[0] / pixelWidth) * 2 - 1) * w
    y = ((screenPos[1] / pixelHeight) * 2 - 1) * w
    f = farClipPlane
    n = nearClipPlane
    z = (screenPos[2] * (f + n))/(f-n) - (2 * (f * n)) / (f - n)
    clipSpace = [x, y, z, w]
    
    return clipSpace

# Converts a time in seconds to frame index
def convert_time_to_frame(event_time, fps=30):
    
    frame_index = int(event_time * fps)
    return frame_index

# Taken from yolov5/utils/general.py
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes



def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    # Convert prediction to tensor
    prediction = np.expand_dims(prediction, axis=0)
    prediction = torch.from_numpy(prediction)

    # device = prediction.device
    device = None
    # mps = 'mps' in device.type  # Apple MPS
    # if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
    #     prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[1] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        class_confidences = x[:, 5:]# .detach().clone()
        object_confidence = x[:,4]# .detach().clone()
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]
            class_confidences = class_confidences[conf.view(-1) > conf_thres]
            object_confidence = object_confidence[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue

        class_confidences = class_confidences[x[:, 4].argsort(descending=True)[:max_nms]]
        object_confidence = object_confidence[x[:, 4].argsort(descending=True)[:max_nms]]
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        
        output[xi] = torch.cat((output[xi],object_confidence[i].unsqueeze(1),class_confidences[i]),dim=1)

        



        # if mps:
        #     output[xi] = output[xi].to(device)
        # if (time.time() - t) > time_limit:
        #     LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        #     break  # time limit exceeded

    return output

# Parse a row of the CSV file
#  We are only interested in a few rows - so filter them here.
def parse_gt_row(row):

    frame_index = convert_time_to_frame(float(row[1]))
    take_number = int(row[2].split("take")[1].strip())
    event_name = row[3].split()[0]

    camera_id = -1
#     if len(row) > 4: # Check which camera this is in
#         camera_id = int(row[4].split("WatchBox")[1][0])
    
#         print(row[4].split("WatchBox")[1])
#         time.sleep(10)

    
        
    return take_number, frame_index, event_name, camera_id

# Here we parse the ground truth file into a specific data structure
# We need to get the following data structure:
#  { ce_number : [(take_number, [ae_name, ae_name, etc],[frame_index, frame_index, etc]), ....] }
#  There are several possible CE numbers:
#   CE1, CE2, CE3, NCE, ICE1, ICE2, etc - the ICE means incomplete CEs, and NCE means non-CEs.
def parse_gt_log(file, take_num):

    # Read in our file
    relevant_frames = []
    with open(file, "r") as rfile:
        gt_reader = csv.reader(rfile, delimiter=',')
        
        # Iterate through every row of the CSV
        for row in gt_reader:

            if not row:  # Seems to stop early
                break

            # If this is the first row for this take, then we just track the title, e.g. "Start Scenario CE3DefensivePosition"
            take_number, frame_index, event_name, camera_id = parse_gt_row(row)
            
            if take_number == take_num:# and camera_id > 0:
                relevant_frames.append((frame_index, event_name))
    
    # No more parsing, just return the results
    return relevant_frames

def setup_socket():


	
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(5)

	
    return client_socket


# def setup_zmq(ip_from_server,ip_to_server,port_from_server,port_to_server):

#     ctx = zmq.Context()

#     #server_ip = zmq_address  # "10.0.0.10" #"10.0.0.10" #"192.168.0.133"#"192.168.0.100" #"192.168.0.133" #"127.0.0.1"

#     #port_from_server = "5561"
#     #port_to_server = "5560"
#     address_from_server = "tcp://%s:%s" % (ip_from_server, port_from_server)
#     address_to_server = "tcp://%s:%s" % (ip_to_server, port_to_server)


#     sock_from_server = ctx.socket(zmq.SUB)
#     sock_from_server.connect(address_from_server)
#     print("address_from_server:", address_from_server)
#     sock_to_server = ctx.socket(zmq.PUB)
#     sock_to_server.connect(address_to_server)
#     sock_from_server.setsockopt(zmq.SUBSCRIBE, b'')
#     print("address_to_server: ", address_to_server)
    
#     return sock_from_server, sock_to_server




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
def watchbox(tracks,state, watchboxes, min_history = 8):
    results = []

    for t_key in tracks.keys():

            # Make sure that there are more than x amount of history before recognizing.
            # print(t_key)
            # print(tracks[t_key][3])
            # if len(tracks[t_key][3]) < min_history:
            #     continue
            

            # if t_key == 13:
            #     print(tracks[t_key][2])
            #     print(tracks[t_key][3])
            #     print(tracks[t_key][3][-1])
            #     input()
                # Also check the most recent history

            select = np.where(watchboxes[:,4] == tracks[t_key][2])[0]
            # with GT: [40, 515, 103, 545]
            # without GT: [     74.118       513.8      92.082       523.2]
            # print(tracks[t_key][0])
            # print(type(tracks[t_key][0]))
            # asdf
            
            if select.size > 0:
                # print(select)
                reference_point = [tracks[t_key][0][0]+(tracks[t_key][0][2] - tracks[t_key][0][0])/2, tracks[t_key][0][1] + (tracks[t_key][0][3] - tracks[t_key][0][1])/2]
                # print(reference_point)
                p1 = reference_point[0] - watchboxes[select,0] > 0
                p2 = reference_point[1] - watchboxes[select,1] > 0
                p3 = reference_point[0] - watchboxes[select,2] < 0
                p4 = reference_point[1] - watchboxes[select,3] < 0

                # if t_key == 12:
                #     print("p1 p2 p3 p4" + str((p1, p2, p3, p4)))

                ptotal = p1 & p2 & p3 & p4

                try:
                    results_tmp = np.logical_xor(ptotal,state[t_key]['watchbox']['results'][select])
                except:
                    # pdb.set_trace()
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
    

    reference_points = np.array([[tracks[t_key][0][0]+(tracks[t_key][0][2] - tracks[t_key][0][0])/2, tracks[t_key][0][1] + (tracks[t_key][0][3] - tracks[t_key][0][1])/2, tracks[t_key][2], t_key] for t_key in tracks.keys()])
    min_number_vehicles = 3
    
    res_per_track_id = {}
    results_tmp = {}    
    for g_idx,g in enumerate(groups):
    
        rp_idxs = np.where(reference_points[:,2] == g[1])[0] #Check error here
        
        if len(reference_points) > min_number_vehicles and len(rp_idxs) > min_number_vehicles:
            
            
            clustering = None # AgglomerativeClustering(n_clusters=None, distance_threshold=g[0], linkage='single').fit_predict(reference_points[rp_idxs,:2])

            labels,label_counts = np.unique(clustering, return_counts=True)
            #pdb.set_trace()
            count_res = np.where(label_counts > min_number_vehicles)[0]

            if count_res.size > 0:
                for possible_convoy in count_res:
                    track_idxs = np.where(clustering == labels[possible_convoy])[0]
                    convoy_elements = reference_points[track_idxs,3]
                    if state[convoy_elements[0]]['convoy']['results'][g_idx]:
                        
                        if not set(state[convoy_elements[0]]['convoy']['results'][g_idx]) == set(convoy_elements):
                            
                            for t_key in tracks.keys():
                                #state[t_key]['convoy']['data'] = convoy_elements.tolist()
                                
                                
                                #print(t_key, "changed to True modified")
                                
                                
                                state[t_key]['convoy']['results'][g_idx] = convoy_elements.tolist()
                                if t_key not in results_tmp:
                                    results_tmp[t_key] = []
                                    
                                results_tmp[t_key].append([g_idx, convoy_elements.tolist()])
                                #print(convoy_elements)
                    else:
                        for t_key in tracks.keys():
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
    

class SensorEventDetector:
    
    
    def sendMessage(self, message, serverAddr):
        
        # Turn the message into bytes
        message = str(message).encode()
        print("sending to " + str(serverAddr))
        # self.sock.sendto(message, serverAddr)
        self.sock.send(message)
        
        
    
    # Perform the handshake
    #  This means first sending this event detector's camera ID
    #  Then listening to the CE server to get the watchbox coordinates
    def handshake(self, currentAddr, serverAddr):
        
        # First, initialize our socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # self.sock.bind(currentAddr)
        # print("Listening socket on " + str(currentAddr))
        self.sock.connect(serverAddr)
        
        # Next, send data to the server address
        camera_message = "camera_id:"+str(self.camera_id)
        self.sendMessage(camera_message, serverAddr)
        

        # Now, receive video data
        print("Receiving video data...")
        video_filepath = "../video_to_run.mp4"
        video_file = open(video_filepath, "wb")
        while True:
            data = self.sock.recv(1024)
            if data == b"done":
                break
            video_file.write(data)

        print("Finished receiving...")
        # Now, send an ack back that you have received, loaded the video, and did warmup
        self.reader = Reader(video_filepath)
        self.pixel_width, self.pixel_height = \
        self.reader.read().shape[1], self.reader.read().shape[0]

        #  Do a warmup for the model
        print("Starting Warmup")
        warmup = np.random.randint(0, 255, self.reader.read().shape).astype("float")
        for i in tqdm(range(10)):
            self.yolo_model(warmup)


        # Send 'ready'
        print("Sending Ready")
        self.sendMessage("ready", serverAddr)

        
        # Next, temporarily listen for data
        data = self.sock.recv(1024)
        
        # Set our watchbox data
        watchbox_data = data.decode()
        print(watchbox_data)
        watchbox_data = watchbox_data.split("watchboxes:")[1]
        watchbox_data = eval(watchbox_data)
        print("received watchbox data: " + str(watchbox_data))
        
        # If we don't get any watchbox data, then we don't call any additional functions
        if watchbox_data:
            self.functions = ["watchbox"]
            self.function_metadata['watchbox'] = np.array(watchbox_data)
        
    # This just tells the server that this camera process has completed
    def completed(self, serverAddr):
        
        self.sendMessage("quitting:", serverAddr)
        self.sock.shutdown(SHUT_RDWR)
        self.sock.close()
        # self.debug_file.close()  # Close our writing file
        
    
    def __init__(self, \
                     yolo_model, track_alg, camera_id, currentAddr, serverAddr,\
                    relevant_frames, result_dir, recover_lost_track, \
                    buffer_zone, ignore_stationary, use_gt):
        
        # Used in tracking
        self.track_alg = track_alg
        self.tracks = {}
        self.state = {}
        self.track_id = 0
        self.old_tracks = []

        # Determine if we are using ground truth.  If so, we can ignore parts of the detection
        #  and tracking.
        self.use_gt = use_gt
        self.gt_mapping = {}  #this maps a special ID to our numbers
        # if self.use_gt:
        #     self.gt_source = gt_source_drone(video_file)
        
        # Determine relevant frames:
        self.relevant_frames = relevant_frames
        self.result_dir = result_dir
        
        # Used to track debug data
        # self.debug_output = {"detections":[], "events":[], "tracks":[]}
        # Get files for us to put stuff into
        debug_filename = '/'.join([self.result_dir, "ae"+str(camera_id)+".txt"])
        # self.debug_file = open(debug_filename, "w", buffering=1)
        
        # Filter classes
        filter_classes = [0.0, 1.0]
        
        
        # Used to record functions
        self.functions = []
        self.function_metadata = {}
        
        # Self explanatory
        self.camera_id = camera_id
        self.yolo_model = yolo_model
        
        # This is where our full pipeline results sit (e.g. watchbox detections)
        self.write_to_file_res = []
        
        # Store data related to our video capture
        #  What frame are we on?  
        self.current_cap_frame_index = -1
        
        
        # Now, initialize our video capture
        # if video_file:
            # self.cap = cv2.VideoCapture(video_file)
            # self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # self.pixel_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # self.pixel_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            # if self.use_gt:
            #     self.gt_source.generate_cam_dict(self.pixel_width, self.pixel_height)

            
            
            

#             if args.metadata:
#                 f_metadata = open(args.metadata)
#                 metadata = json.load(f_metadata)

        # else:
        #     client_socket = setup_socket()
        #     client_socket.connect((address, port))
        #     sock_from_server, sock_to_server = setup_zmq(args.address_from_server,args.address_to_server,args.port_from_server,args.port_to_server)
        #     pixel_width = 800
        #     pixel_height = 600
        #     imgsz = pixel_width*pixel_height*4
            
        # Also initialize our trackers
        #Choose tracking algorithm
        self.tracker = None
        if track_alg == 'Byte':
            # My buffer size is X minutes at 30fps - if a track is lost for more than X minutes, it is deleted.
            new_args = ByteTrackArgs(0.5,0.8, 10000000)  # Basically always track same ID
            self.tracker = BYTETracker(new_args, filter_classes, recover_lost_track, \
                buffer_zone, ignore_stationary, frame_rate=10)
        elif track_alg == 'Sort':
            self.tracker = Sort(new_args.track_thresh)
        elif track_alg == 'MOTDT':
            self.tracker = OnlineTracker('trackers/pretrained/googlenet_part8_all_xavier_ckpt_56.h5', use_tracking=False)
        elif track_alg == 'DeepSort':
            self.tracker = DeepSort('trackers/pretrained/ckpt.t7')
            
            
        # Lastly, be sure to set up our server stuff for this detector
        self.currentAddr = currentAddr
        self.serverAddr = serverAddr
        self.sock = None
        

        # do the handshake
        self.handshake(currentAddr, serverAddr)

        

        

        # Configs for nms
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000

        
        
    
    # Do a prediction using the model and NMS
    def do_prediction(self, image):

        # Predict using the model
        output = self.yolo_model(image)

        # Now run nms
        pred = non_max_suppression([output], self.conf_thres, self.iou_thres, \
            self.classes, self.agnostic_nms, max_det=self.max_det)
        
        line_return = []
        # Process predictions
        start_process_time = time.time()
        # print(pred[0])
        for i, det in enumerate(pred):  # per image

            im0, frame = image.copy(), 0

            # s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if self.save_crop else im0  # for self.save_crop
            # annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # print("here")
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for det_row in reversed(det):

                    xyxy = det_row[:4].cpu()
                    conf = det_row[4].cpu()
                    cls = det_row[5].cpu()
                    extra = det_row[6:].cpu()


                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf, *extra) # if self.save_conf else (cls, *xywh)  # label format
                    line_return.append(('%g ' * len(line)).rstrip() % line)

        return line_return
    

    # Grab an image at an index, and get the yolo results
    def do_object_detection(self, frame, frame_index):
        
        start_read_time = time.time()
        # On our first frame capture, we use set
        # image = None
        # if self.current_cap_frame_index < 0:
        #     self.cap.set(1, frame_index)
        #     ret, image = self.cap.read()
        # else:  # Otherwise, we capture from the current frame index to the current goal frame
        #     frame_count_diff = frame_index - self.current_cap_frame_index
        #     for i in range(frame_count_diff):
        #         ret, image = self.cap.read()
        # Be sure to set the new frame index
        # self.current_cap_frame_index = frame_index

        res_lines = []


        if self.use_gt:  # Get the ground truth res lines
            res_lines = self.gt_source.generate_results_for_frame(self.camera_id, frame_index)
        else:
            res_lines = self.do_prediction(frame)
            # print(len(res_lines))
            # print(res_lines[0])
            # res_lines = self.yolo_model.run(image) #Run yolo
        
        return frame, res_lines

    # Check if a newly assigned track is allowed
    def allowed_track_entrance(self, bbox):
        allowed = False

        if bbox[0] > 0 and bbox[1] > 0 and bbox[2] < 300 and bbox[3] < 1079:
            allowed = True

        return allowed

        
    # Now execute our tracker
    def update_tracker(self, image, res_lines, frame_index):
        
        detection_bboxes = np.array([])
        detection_class_ids = np.array([])
        detection_confidences = np.array([])
        detection_extra = np.array([])

        # So here's the content of each element of res_lines:
        #  [ class prediction, x1, y1, x2, y2, objectness score, something, then class scores ]

       
        # Iterate through every YOLO result, and add them to detection_bboxes
        for line in res_lines:


            coordinates_line = line.split()

            if int(coordinates_line[0]) > 2: #Only pedestrians
                continue

            box_voc = None
            if self.use_gt:
                box_voc = coordinates_line[1:5]
                box_voc = [int(x) for x in box_voc]
                class_id = int(coordinates_line[0])
                #  Get the class and object ID
                # Create our tracks here.
                obj_id = coordinates_line[5]
                if obj_id not in self.gt_mapping:
                    self.gt_mapping[obj_id] = self.track_id
                    self.tracks[self.track_id] = (np.array(box_voc), frame_index, class_id)
                    
                    if self.track_id not in self.state:
                        self.state[self.track_id] = {}
                        self.state = state_init(self.state,self.track_id,self.functions,self.function_metadata)

                    self.track_id += 1

                else:
                    current_id = self.gt_mapping[obj_id]
                    self.tracks[current_id] = (np.array(box_voc), frame_index, class_id)

                    # Now, put text on the image
                    fontScale = 0.5
                    color = (255, 153, 255)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    thickness = 2
                    image = cv2.putText(image, str(current_id), (box_voc[0],box_voc[1]), font, 
                                fontScale, color, thickness, cv2.LINE_AA)
                

            else:    
                box_voc = pbx.convert_bbox((float(coordinates_line[1]),float(coordinates_line[2]),float(coordinates_line[3]),float(coordinates_line[4])), from_type="yolo", to_type="voc", \
                                            image_size=(self.pixel_width, self.pixel_height))

                if detection_bboxes.size == 0:
                    detection_bboxes = np.expand_dims(np.array(box_voc),axis=0)
                else:
                    detection_bboxes = np.concatenate((detection_bboxes, np.expand_dims(np.array(box_voc),axis=0)),axis=0)
                #detection_bboxes = np.append(detection_bboxes, np.expand_dims(np.array(box_voc),axis=0),axis=0)
                detection_class_ids = np.append(detection_class_ids, int(coordinates_line[0]))
                detection_confidences = np.append(detection_confidences, float(coordinates_line[5]))

                extra_data = np.array([float(cl) for cl in coordinates_line[6:]])
                if detection_extra.size == 0:
                    detection_extra = np.expand_dims(extra_data,axis=0)
                else:
                    detection_extra = np.concatenate((detection_extra,np.expand_dims(extra_data,axis=0)),axis=0 )


            # Finally, draw the rectangle
            cv2.rectangle(image, (box_voc[0], box_voc[1]), (box_voc[2], box_voc[3]), (0, 0, 255), 1)



        #print(detection_bboxes)
        #o_tracks = tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
        #pdb.set_trace()

        text_output = ''
        issue_pause = False
        if detection_bboxes.size > 0:

            if self.track_alg == 'MOTDT' or self.track_alg == 'DeepSort':
                online_targets, issue_pause = \
                    self.tracker.update(np.column_stack((detection_bboxes,detection_confidences)), \
                                [self.pixel_height, self.pixel_width], (self.pixel_height, self.pixel_width), image2)
            else:
                bbox_stack = np.column_stack((detection_bboxes, detection_confidences))
                online_targets, issue_pause = \
                    self.tracker.update(bbox_stack, \
                                        [self.pixel_height, self.pixel_width], (self.pixel_height, self.pixel_width),detection_class_ids, image, detection_extra)

            new_tracks = []
            #pdb.set_trace()
            for t_idx,t in enumerate(online_targets):

                # print(online_targets)
                # input()

                # if track_alg == 'Sort' or track_alg == 'DeepSort':
                #     track_id = int(t[4])
                #     bbox = t[:4]
                # else:
                self.track_id = t.track_id
                bbox = t.tlbr
                class_history = t.detected_class
                detection_extra = t.detected_extra
                new_tracks.append(self.track_id)


                
                

                # Check something - if this track does not already exist and did not emerge
                #   from one side of the screen, then we ignore this new track
                # print(bbox)
                # print(class_history)
                # print(self.track_id)
                # if len(class_history) <= 1 and not self.allowed_track_entrance(bbox):
                #     continue
                # input()

                # print("Allowed over...")
                


                    

                class_detected = t.voted_class

                # if self.track_id == 266:
                #     print(class_history[-1])
                #     print(class_detected)
                #     print()
                #     input()


                self.tracks[self.track_id] = (bbox,frame_index,class_detected, \
                    class_history, detection_extra)
                
                
                if self.track_id not in self.state:
                    self.state[self.track_id] = {}
                    self.state = state_init(self.state,self.track_id,self.functions,self.function_metadata)

                #Put label outside bounding box
                fontScale = 0.5
                color = (255, 153, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                image = cv2.putText(image, str(self.track_id), (int(bbox[0]),int(bbox[1])), font, 
                               fontScale, color, thickness, cv2.LINE_AA)

#                 if args.save_tracking_dir:
#                     text_output += "%f,%f,%f,%f,%d,%d\n" % (*bbox,track_id, class_detected)


            if set(new_tracks) != set(self.old_tracks):
               
                for tt in list(set(self.old_tracks) - set(new_tracks)):
                    del self.tracks[tt]


            self.old_tracks = new_tracks

#             if args.save_tracking_dir:
#                  f = open(args.save_tracking_dir + '/' + str(f_idx) + '.txt', "w")
#                  f.write(text_output)
#                  f.close()

#             if args.create_video:
#                 video.write(image)

        
        return image, issue_pause
    
    
    # Execute the additional functions, such as watchbox recognition, etc.
    def execute_additional_functions(self, image, frame_index):
        
        frame_result = []
        for f in self.functions:
            #Apply functions according to query (right now only two tripwires are checked)
            res,state = eval(f+"(self.tracks,self.state,self.function_metadata['" + f +"'])")
            
            if f =='watchbox':
                 # Also, add the bounding boxes to check if this is working
                for wb in self.function_metadata["watchbox"]:
                    cv2.rectangle(image, (wb[0], wb[1]), \
                                      (wb[2], wb[3]), (0, 0, 255), 1)
            
            if res:
                new_res = {}
                new_res["camera_id"] = self.camera_id
                new_res["results"] = res
                new_res['time'] = frame_index

                if f == 'watchbox':
                    print("Watchbox entered or not")
                    print(new_res)
                    # self.write_to_file_res.append(new_res)
                    frame_result.append(new_res)
                    #sock_to_server.send_multipart([f.encode("utf-8"), pickle.dumps(new_res)])
                    
                   


                if f == 'cross_tripwire': #If there is an event (change of state)
                    print("Tripwires", r[0], "crossed by", r[1], "at", frame_index+1)
                    #sock_to_server.send_multipart([topic.encode("utf-8"), pickle.dumps(events_by_topic[topic])])
                    #pdb.set_trace()
                    #screen_coordinates = ScreenToClipSpace((float(coordinates_line[1])*pixel_width, float(coordinates_line[2])*pixel_height, camera_height),pixel_width,pixel_height,far_clip_plane, near_clip_plane)
                    #world_coordinates = np.matmul(inv_matrix,screen_coordinates)

                if f == 'convoy':
                    m, s = divmod(frame_index/fps, 60)
                    h, m = divmod(m, 60)
                    print("Convoy", new_res, m,s)
                    
        return image, frame_result

    
    # Execute our full detection pipeline, from yolo to tracking to watchbox processing
    def execute_full_detection(self, frame_index, frame, stride):
            
        start_loop_time = time.time()
        
        data_out = [frame_index]

        # Get the object detection result
        image, res_lines = self.do_object_detection(frame, frame_index)
        # print("Time for object detection: %f" % (time.time()-start_loop_time))


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
        # image = cv2.putText(image, 'Traffic Camera ID: ' + str(self.camera_id), org, font, 
        #            fontScale, color, thickness, cv2.LINE_AA)


        #pdb.set_trace()
        #res = open('exp5/labels/out-%04d.txt' %(f_idx+1))
        #res_lines = res.readlines()



        #time_past = time.time()
        issue_pause = False
        if res_lines:
            # self.debug_output["detections"].append((res_lines, frame_index))
            data_out.append(res_lines)
            
            # Now, update our tracker
            track_start_time = time.time()
            image, issue_pause = self.update_tracker(image, res_lines, frame_index)

#             out_track_data = self.tracks
#             for tkey in out_track_data.keys():
#                 out_track_data[tkey] = list(out_track_data[tkey])
                
#                 out_track_data[tkey][0] = list(out_track_data[tkey][0])
#                 out_track_data[tkey] = str(out_track_data[tkey])
                
                
            # self.debug_output["tracks"].append((self.tracks, frame_index))
            data_out.append(self.tracks)
            # print("Time for BYTE tracker: %f" % (time.time()-track_start_time))



        # Perform extra processing, such as for tripwires of watchboxes
        func_start_time = time.time()
        image, frame_result = self.execute_additional_functions(image, frame_index)
        # # self.debug_output["events"].append((frame_result, frame_index))
        data_out.append(frame_result)
        
        # print("Time for additional functions: %f" % (time.time()-func_start_time))
        
        # Now, if we actually have results, be sure to send it back to the server
        # if frame_result:
        #     self.sendMessage(frame_result, self.serverAddr)
        #     # When we send something, be sure to listen for a message back
        #     # returned_data, addr = self.sock.recvfrom(512)
        #     # returned_data = returned_data.decode()
            
        #     # If our returned data is somehting other than 'none':
        #     # if returned_data != "none":
        #     # First, create a folder for ce_images
        #     filepath = [self.result_dir, "ce_images"]
        #     if not os.path.exists('/'.join(filepath)):
        #         os.mkdir('/'.join(filepath))
        #     filename = "frame"+str(frame_index)+"_cam"+str(self.camera_id) + ".jpg"
        #     filepath.append(filename)
        #     cv2.imwrite('/'.join(filepath), image)
                
        # Remember the image data for ground truth times
        # for rframe in self.relevant_frames:
        #     if abs(frame_index - rframe[0]) < stride:
                
        #         # ground truth folder
        #         filepath = [self.result_dir, "gt_images"]
        #         if not os.path.exists('/'.join(filepath)):
        #             os.mkdir('/'.join(filepath))
                
        #         # Save the image
        #         filename = "frame"+str(frame_index)+"_cam"+str(self.camera_id) + "_ev" + rframe[1] + ".jpg"
        #         filepath.append(filename)
        #         cv2.imwrite('/'.join(filepath), image)

        # cv2.imshow('image'+str(self.camera_id),image)
        # cv2.waitKey(1)

        if issue_pause:
            print("Pausing")
            # input()
        
        # Write data out to file
        # self.debug_file.write(':::'.join([str(x) for x in data_out]) + "\n")
        


        # print("Total exec time:", time.time()-start_loop_time)

        



#ce_json = open("complex_results.json","w")
#json.dump(write_to_file_res,ce_json)

# if args.create_video:
#     video.release()
    
    
#source = '../../Delivery-2022-12-12/video1/'
#files = sorted(glob.glob(os.path.join(source, '*.*')))


#Metadata with camera information
#f = open('../../Delivery-2022-12-12/metadata.json')

#metadata = json.load(f)



fps = 10

#Two different tripwires defined as the location with respect to the image width for a given pixel column
#tripwire1 = 1300
#tripwire2 = 1750
# functions = ['convoy']
# function_metadata['convoy'] = [[80,0]]

# write_to_file_res = []

#new_function = {'watchbox':function_metadata['watchbox']}
#state = state_add(state,['watchbox'],new_function)

parser = argparse.ArgumentParser(description='Edge Processing Node')
# parser.add_argument('--address', type=str, help='Address to connect to receive images')
# parser.add_argument('--port', type=int, help='Port to connect to receive images')
# parser.add_argument('--address_to_server', type=str, help='Address to connect to send atomic events')
# parser.add_argument('--port_to_server', type=int, help='Port to connect to send atomic events')
# parser.add_argument('--address_from_server', type=str, help='Address to connect to receive topic subscriptions')
# parser.add_argument('--port_from_server', type=int, help='Port to connect to receive topic subscriptions')
# parser.add_argument('--camera_id', type=int, help='Camera id')
parser.add_argument('--track_alg', type=str, default='Byte', help='Track algorithm: Byte, Sort, DeepSort or MOTDT')
# parser.add_argument('--video-file', type=str, default='', help='Open video file instead of connecting to server')
parser.add_argument('--weights', type=str, default='./yolov5s.engine', help="YOLO weights file")
# parser.add_argument('--device', type=str, default='0', help="Device where to run YOLO")
# parser.add_argument('--metadata', type=str, help='Camera metadata file')
# parser.add_argument('--save-tracking-dir', type=str, default='', help='Save tracking results to the directory specified')
# parser.add_argument('--create_video', type=str, default='', help='Create video. Specify file name.')
# parser.add_argument('--yolo-synth-output', type=str, default='', help='Provide YOLO files and bypass the model. Specify the directory name.')

# # New stuff...
# parser.add_argument('--video_files', nargs='+', help='Video files.', required = True)
# parser.add_argument('--camera_ids', nargs='+', help='Corresponding camera ids to each video file.', required = True)
# parser.add_argument('--start_port', type=int, required = True)
# parser.add_argument('--server_port', type=int, required = True)
# parser.add_argument('--current_take', type=int, required = True)
# parser.add_argument('--result_dir', type=str, required = True)
# parser.add_argument('--recover_lost_track', action='store_true')  # If our tracker keeps lost tracks
# parser.add_argument('--no_recover_lost_track', action='store_true')  # If our tracker keeps lost tracks
# parser.add_argument('--buffer_zone', type=int, required = True)
# parser.add_argument('--ignore_stationary', action='store_true')  # If our tracker keeps lost tracks
# parser.add_argument('--no_ignore_stationary', action='store_true')  # If our tracker keeps lost tracks
# parser.add_argument('--no_use_gt', action='store_true')
# parser.add_argument('--use_gt', action='store_true')  # If we use the ground truth

# parser.add_argument('--ce', type=int, help='Determines which CE we want to capture')
# parser.add_argument('--server_port', type=int, help='Determines which CE we want to capture')
# parser.add_argument('--result_dir', type=str, help='this is where we will output our json')
# parser.add_argument('--debug_output', type=str, help='this is where we will output debug data')



args = parser.parse_args()





# address = args.address
# port = args.port


# Temporarily open the video capture to check dimensions
# cap = cv2.VideoCapture(args.video_file)
# pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cap.release()


# if not args.yolo_synth_output:
    # yolo = Yolo_Exec(weights=args.yolo_weights, imgsz=[1920],conf_thres=0.5, device=args.device, save_conf=True) #'../../delivery-2022-12-01/t72detect_yv5n6.pt',imgsz=[2560],conf_thres=0.5)
yolo = Yolov5Trt(classes="coco",
                    backend="tensorrt",
                    weight=args.weights,
                    auto_install=False,
                    dtype="fp16")


# last_message_num = 0

# first_pass = True
#tracker = SORT(max_lost=20, iou_threshold=0.0001, tracker_output_format='mot_challenge')
#tracker = CentroidKF_Tracker(max_lost=20)

# decode_message = False



class ByteTrackArgs:
    def __init__(self, track_thresh, match_thresh, track_buffer):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.mot20 = False 
        self.track_buffer = track_buffer







#Create video file
# if args.create_video:
#     video = cv2.VideoWriter(args.create_video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (pixel_width,pixel_height))


if __name__=='__main__':
    
    try:
    #Create directory for saving tracking results
    # if args.save_tracking_dir:
    #     if not os.path.exists(args.save_tracking_dir):
    #         os.makedirs(args.save_tracking_dir)
    
    
    # I suspect the 4th element of this watchbox metadata is for matching classes...
    #  Class 1.0 is the BTR80

        
        # recover_lost_track = False
        # if args.recover_lost_track:
        #     recover_lost_track = True
        # ignore_stationary = False
        # if args.ignore_stationary:
        #     ignore_stationary = True


        event_detectors = []
        # video_files = args.video_files
        # print(video_files)
        # print(args.camera_ids)
        
        video_file = "../video.mp4"
        camera_id = 0


        # Next, also be sure to register the frames of interest
        # ce_file = "../neuroplexLog.csv"
        # This is of the structure   { ce_number : [(take_number, [ae_name, ae_name, etc],[frame_index, frame_index, etc]), ....] }
        # relevant_frames = parse_gt_log(ce_file, args.current_take)


        # Initialize all of our detectors for each video file
        event_detector_ip = "127.0.0.1"
        event_detector_ip = "192.168.55.1"
        # event_detector_port = args.start_port
        event_detector_port = 6703
        ce_server_ip = "127.0.0.1"
        ce_server_ip = "192.168.55.100"
        # ce_server_port = args.server_port
        ce_server_port = 6792
        SERVER_ADDR = (ce_server_ip, ce_server_port)

        result_dir = "results"
        recover_lost_track = False
        buffer_zone = 20
        ignore_stationary = False
        use_gt = False
        relevant_frames = None
        # for v_i, vfile in enumerate(video_files):

            # We only initialize for camera 3
            # if "cam2" not in vfile: # or "cam2" not in vfile:
            #     continue

        # First, initialize the class



        while True:
            eventDetector = SensorEventDetector(yolo, args.track_alg, camera_id, \
                                                (event_detector_ip, event_detector_port),
                                                (ce_server_ip, ce_server_port), relevant_frames, result_dir, \
                                                    recover_lost_track, buffer_zone, \
                                                    ignore_stationary, use_gt)

            # event_detector_port += 1
            # event_detectors.append(eventDetector)


            # total_frames = eventDetector.total_frames
            start_frame = 0 # int(1800*5) # int(1800*4.35)
            stride = 3  # Speed up our execution - we skip every few frames

            # Make sure we have a directory to save some images - basically when events happen 
            # This should be for both ground truth frames () and when this detector detects them.


            # Now, iterate over all frames 
            last_sample_time = time.time()
            num_frames = 0
            frame_index = 0
            # for frame_index in tqdm(range(start_frame, total_frames, stride)):

            for frame in tqdm(eventDetector.reader):

                # for eventDetector in event_detectors:
                    # Then we can run the full detection pipeline
                eventDetector.execute_full_detection(frame_index, frame, stride)
                
                if time.time() - last_sample_time > 1:
                    # print("Processing rate: %d fps (per camera)" % (num_frames))
                    num_frames = 0
                    last_sample_time = time.time()
                num_frames += 1
                frame_index += 1

            # Next, iterate again over all the eventDetectors to print their results to a file

            # for e_i, eventDetector in enumerate(event_detectors):
                # eventDetector.cap.release()
                # We are quitting...
            eventDetector.completed(eventDetector.serverAddr)
                # print(e_i)
                # Write our results to a json file
                # result_file = '/'.join([args.result_dir, "ae"+str(args.camera_ids[e_i])+".json"])
                # with open(result_file, "w") as wfile:
                #     json.dump(eventDetector.debug_output, wfile)
                    




            # Lastly, specifically add a tag if this executed correctly
            print("Ended correctly.")
            time.sleep(2)
    
    except Exception as e:
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        print(traceback.format_exc())
        input()


    

    # Todo: 
    #  You should try to export this YOLO model into tensorRT, otherwise it'll be way too slow.
    #  THe FPS seems to be used in the tracker - is it important to set correctly?