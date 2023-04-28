from ce_builder import sensor_event_stream, watchbox, complexEvent, Event, OR, AND, GEN_PERMUTE, WITHIN
import time
import os
import cv2
import numpy as np

import json
import socket
import threading
import argparse

import traceback

# data_file = open("complex_results.json", "r")
# json_results = json.load(data_file)
# data_file.close()

def get_data(frame_index):
    data = []
    
    for x in json_results:
        if x['time'] == frame_index:
            data.append(x)
    return data


# Send data - encode it 
def sendMessage(message, addr, sock):
    # Turn the message into bytes
    message = str(message).encode()
    print("sending to " + str(addr))
    sock.sendto(message, addr)
        

# This is our listening server, which is where we get all of our results
def server_listen(ce_obj, ce_structure):

    try:
        # First, bind the server
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        serverSocket.bind(SERVER_ADDR)
        
        # Open our file for writing
        debug_filename = '/'.join([ce_obj.result_dir, "ce_output.txt"])
        debug_file = open(debug_filename, "w", buffering=1)
        
        print("Listening on " + str(SERVER_ADDR))

        # Send data to the fsm server
        time.sleep(2)
        sendMessage(ce_structure, ("127.0.0.1", 8001), serverSocket)
        print("HERE")
    except Exception as e:
        print(e)
        input()
    


    # Now for the recv logic
    while True:
        try:
            data, addr = serverSocket.recvfrom(512)
            if data:
                decoded = data.decode()
                # If this is a handshake message containing camera id
                if "camera_id:" in decoded:
                    cam_id = int(decoded.split(":")[1])
                    ce_obj.client_info[cam_id] = addr
                    print(cam_id)
                    print(ce_obj.config_watchboxes)
                    # Be sure to hand back the corresponding watchboxes
                    if cam_id in ce_obj.config_watchboxes:
                        return_message = "watchboxes:" + str(ce_obj.config_watchboxes[cam_id])
                    else:
                        return_message = "watchboxes:[]"
                    print(return_message)
                    sendMessage(return_message, addr, serverSocket)
                elif "quitting:" in decoded:

                    print("quitout!")

                    # Write our data to file
                    # result_file = '/'.join([ce_obj.result_dir, "ce_output.json"])
                    # with open(result_file, "w") as wfile:
                    #     json.dump(ce_obj.result_output, wfile)
                    # break
                    debug_file.close()
                    break
                else:

                    # Format is like: [{'camera_id': '2', 'results': [[[0], [True], 3]], 'time': 17910}]
                    
                    # Otherwise, this is data.
                    incoming_data = eval(decoded)
                    print("Receiving " + str(incoming_data))
                    
                    
                    # Get the timestamp of this event:
                    frame_index_incoming = incoming_data[0]["time"]
                    data_to_write = [frame_index_incoming]
                    
                    
                    # ce_obj.result_output["incoming"].append([incoming_data, frame_index_incoming])
                    data_to_write.append(incoming_data)


                    ce_obj.update(incoming_data)
                    result, change_of_state, old_results = ce_obj.evaluate(frame_index_incoming)

                    # ce_obj.result_output["events"].append([\
                    #                                       result, \
                    #                                       change_of_state, \
                    #                                       old_results, \
                    #                                       frame_index_incoming])
                    for res in result:
                        data_to_write.append([res, change_of_state, old_results])
                    
                        debug_file.write(":::".join([str(x) for x in data_to_write]) + "\n")
                        data_to_write = data_to_write[:-1]  # Pop the last element back out
                    
                    # Check if any event became true
                    # message_to_send = "none"
                    # if len(result) >  and result[3]:
                    #     event_turned_true = result[0][0]
                    #     message_to_send = event_turned_true
                    # Send this message back 
                    # sendMessage(message_to_send, addr, serverSocket)
                        

                    if change_of_state:
                        print()
                        print(old_results)
                        print(result)
                        # Send data to our fsm display server
                        sendMessage((old_results, result), ("127.0.0.1", 8001), serverSocket)
                        
                        
                    
        except Exception as e:
            print(traceback.format_exc())
            input()
    
    debug_file.close()


def build_ce1():
    # CE1
    # First, initialize our complex event
    ce1 = complexEvent()
    ce_structure = []
    # Set up our watchboxes
    ce1.addWatchbox("bridgewatchbox0 = watchbox(camera_id='2', positions=[200,1,1919,1079], watchbox_id=0)")
    ce1.addWatchbox("bridgewatchbox1 = watchbox(camera_id='2', positions=[213,274,772,772], watchbox_id=1)")
    ce1.addWatchbox("bridgewatchbox2 = watchbox(camera_id='2', positions=[816,366,1200,725], watchbox_id=2)")
    ce1.addWatchbox("bridgewatchbox3 = watchbox(camera_id='2', positions=[1294,290,1881,765], watchbox_id=3)")
    ce1.addWatchbox("bridgewatchbox4 = watchbox(camera_id='1', positions=[413,474,1072,772], watchbox_id=0)")
    ce1.addWatchbox("bridgewatchbox5 = watchbox(camera_id='0', positions=[1,1,1919,1079], watchbox_id=0)")

    # Now we set up the atomic events

    # First, fource vehicles show up in camera 0
    vehicles_head_to_bridge = Event("bridgewatchbox5.composition(at=0, model='rec_vehicle').size==4 and bridgewatchbox5.composition(at=1, model='rec_vehicle').size!=4")

    # Next, four vehicles approach the bridge from one side
    vehicles_approach_bridge = Event("bridgewatchbox0.composition(at=0, model='rec_vehicle').size==4 and bridgewatchbox0.composition(at=1, model='rec_vehicle').size!=4")

    # Next, we have two vehicles on either side of the bridge
    vehicles_plant_bombs = Event( "bridgewatchbox1.composition(at=0, model='rec_vehicle').size==2 and bridgewatchbox3.composition(at=0, model='rec_vehicle').size==2" )

    # And we have more than 4 vehicles in cam2
    tanks_present = Event( "bridgewatchbox4.composition(at=0, model='tank').size>=4" )

    # Then we have two vehicles exit watchbox 3
    vehicles_leave = Event( "bridgewatchbox3.composition(at=1, model='rec_vehicle').size==2 and bridgewatchbox3.composition(at=0, model='rec_vehicle').size==0" )

    # And finally we add these events together
    ce_structure = ce1.addEventSequence([vehicles_head_to_bridge, vehicles_approach_bridge, AND(tanks_present, vehicles_plant_bombs), GEN_PERMUTE(vehicles_leave, "size")])
    # ce_structure = ce1.addEventSequence([vehicles_approach_bridge, vehicles_plant_bombs, GEN_PERMUTE(vehicles_leave, "size")])

    return ce1, ce_structure

def build_ce2():

    ce2 = complexEvent()
    # Set up our watchboxes
    ce2.addWatchbox("bridgewatchbox0 = watchbox(camera_id='2', positions=[1,1,1919,1079], watchbox_id=0)")
    ce2.addWatchbox("bridgewatchbox1 = watchbox(camera_id='2', positions=[213,274,772,772], watchbox_id=1)")
    ce2.addWatchbox("bridgewatchbox2 = watchbox(camera_id='2', positions=[816,366,1200,725], watchbox_id=2)")
    ce2.addWatchbox("bridgewatchbox3 = watchbox(camera_id='2', positions=[1294,290,1881,765], watchbox_id=3)")
    ce2.addWatchbox("roadwatchbox1 = watchbox(camera_id='0', positions=[213,274,1750,765], watchbox_id=0)")
    ce2.addWatchbox("roadwatchbox2 = watchbox(camera_id='1', positions=[1294,274,1881,765], watchbox_id=0)")

    # Now we set up the atomic events

    # First, four recce vehicles move in formation through camera1
    vehicle_move = Event("roadwatchbox1.composition(at=0, model='rec_vehicle').size==4 and roadwatchbox1.composition(at=1, model='rec_vehicle').size<4")

    # Then in camera3, four vehicles approach the bridge from one side
    vec_approach_bridge = Event("bridgewatchbox0.composition(at=0, model='rec_vehicle').size==4 and bridgewatchbox0.composition(at=1, model='rec_vehicle').size<4")

    # Next, we have two vehicles on either side of the bridge
    rec_on_either_side = Event("bridgewatchbox1.composition(at=0, model='rec_vehicle').size==2 and bridgewatchbox3.composition(at=0, model='rec_vehicle').size==2")

    # In camera2, observe tanks moving toward the bridge (e.g. entering wb then leaving)
    tanks_exit_bridge = Event("roadwatchbox2.composition(at=1, model='tank').size>0 and roadwatchbox2.composition(at=0, model='tank').size==0")

    # Either ev21 occurred first, then ev23 within X frames (3 minutes = 5400 frames), or the other way around
    rec_first = WITHIN(rec_on_either_side, tanks_exit_bridge, 5400)
    tank_first = WITHIN(tanks_exit_bridge, rec_on_either_side, 5400) 

    # And finally we add these events together
    #  In this case, we do not enforce a strict sequence, because tanks_exit_bridge can occur before or after rec_on_either_side
    #  We do however enforce the time constraint that one or the other occurs within a certain time
    ce_structure = ce2.addEventSequence([vehicle_move, vec_approach_bridge, rec_on_either_side, tanks_exit_bridge, OR(tank_first, rec_first)], no_enforce_sequence=True )

    return ce2, ce_structure


def build_ce3():

    ce3 = complexEvent()
    # Set up our watchboxes
    
    ce3.addWatchbox("rwatchbox1 = watchbox(camera_id='1', positions=[213,274,772,772], watchbox_id=1)")
    ce3.addWatchbox("rwatchbox2 = watchbox(camera_id='1', positions=[750,366,1200,725], watchbox_id=2)")
    ce3.addWatchbox("rwatchbox3 = watchbox(camera_id='1', positions=[1210,290,1750,765], watchbox_id=3)")
    # Need one watchbox object per type of object
    ce3.addWatchbox("rwatchbox0tank = watchbox(camera_id='1', positions=[200,1,1919,1079], watchbox_id=0)")
    ce3.addWatchbox("r0watchbox0tank = watchbox(camera_id='0', positions=[200,1,1919,1079], watchbox_id=0)")
    ce3.addWatchbox("rwatchbox0rec = watchbox(camera_id='1', positions=[200,1,1919,1079], watchbox_id=4)")
    ce3.addWatchbox("r0watchbox0rec = watchbox(camera_id='0', positions=[200,1,1919,1079], watchbox_id=1)")

    # 3.1 -  recce vehicles in opposite watchbox
    recce_in_opposite_wb = Event("rwatchbox1.composition(at=0, model='rec_vehicle').size==2 and rwatchbox3.composition(at=0, model='rec_vehicle').size==2")

    # 3.0 - recce vehicles and tanks present in camera, for either camera id 1 or camera id 0
    all_in_cam0 = Event("r0watchbox0rec.composition(at=0, model='rec_vehicle').size>=4 and r0watchbox0tank.composition(at=0, model='tank').size>=4")
    all_in_cam1 = Event("rwatchbox0rec.composition(at=0, model='rec_vehicle').size>=4 and rwatchbox0tank.composition(at=0, model='tank').size>=4")

    # 3.2 - tanks in middle watchbox.
    tank_in_middle = Event("rwatchbox2.composition(at=0, model='tank').size>=4")

    # The above events can occur in any order
    ce_structure = ce3.addEventSequence([recce_in_opposite_wb, tank_in_middle, OR(all_in_cam1, all_in_cam0)], no_enforce_sequence=True)

    return ce3, ce_structure


import argparse


parser = argparse.ArgumentParser(description='Edge Processing Node')
parser.add_argument('--ce', type=int, help='Determines which CE we want to capture')
parser.add_argument('--server_port', type=int, help='Determines which CE we want to capture')
parser.add_argument('--result_dir', type=str, help='this is where we will output our json')
parser.add_argument('--debug_output', type=str, help='this is where we will output debug data')
args = parser.parse_args()
SERVER_ADDR = ("127.0.0.1", args.server_port)


if __name__=='__main__':

    try:
        complexEventObj = None
        ce_structure = []
        if args.ce == 1:
            complexEventObj, ce_structure = build_ce1()
        
            complexEventObj.result_dir = args.result_dir
        
            # Our bounding boxes are as follows:
            watchboxes = {
                2: [ [1,1,1919,1079, 1], [213,274,772,772,1], [816,366,1200,725,1], [1294,290,1881,765,1]],
                1: [ [413,274,1072,772, 0] ],
                0: [ [1,1,1919,1079, 1] ]
            }
            complexEventObj.config_watchboxes = watchboxes

        elif args.ce == 2:

            complexEventObj, ce_structure =  build_ce2()
            complexEventObj.result_dir = args.result_dir

            watchboxes = {
                2: [ [1,1,1919,1079,1], [213,274,772,772,1], [816,366,1200,725,1], [1294,290,1881,765,1]],
                1: [ [1294,274,1881,765, 0] ],
                0: [ [1,1,1919,1079, 1] ]
            }
            complexEventObj.config_watchboxes = watchboxes

        elif args.ce == 3:

            complexEventObj, ce_structure =  build_ce3()
            complexEventObj.result_dir = args.result_dir

            # Our bounding boxes are as follows:
            watchboxes = {
                1: [ [1,1,1919,1079, 0], [213,274,772,772,1], [750,366,1200,725,0], [1210,290,1750,765,1], [1,1,1919,1079,1] ],
                0: [ [1,1,1919,1079, 0], [1,1,1919,1079, 1] ]
            }
            complexEventObj.config_watchboxes = watchboxes
            


    except Exception as e:
        print(traceback.format_exc())
        input()
    
    
    # # Set up our server
    server_listen_thread = threading.Thread(target=server_listen, args=(complexEventObj,ce_structure,))
    server_listen_thread.start()

#     if incoming_data:
#         #### RUN OUR EVALUATION ON THE EVENT ANYTIME WE GET NEW DATA
#         ce1.update(incoming_data)

#         event_occurred, results = ce1.evaluate()
#         if event_occurred:

#             for result in results:
#                 print("Event %s has changed to %s at frame %d" %(result[0], str(result[1]), frame_index))

    
    