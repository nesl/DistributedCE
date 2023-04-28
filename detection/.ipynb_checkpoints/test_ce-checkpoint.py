from ce_builder import sensor_event_stream, watchbox, complexEvent, Event, OR, AND, GEN_PERMUTE
import time
import os
import cv2
import numpy as np

import json
import socket
import threading
import argparse



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
def server_listen(ce_obj):
    # First, bind the server
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    serverSocket.bind(SERVER_ADDR)
    
    # Open our file for writing
    debug_filename = '/'.join([ce_obj.result_dir, "ce_output.txt"])
    debug_file = open(debug_filename, "w", buffering=1)
    
    print("Listening on " + str(SERVER_ADDR))
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

                    # Be sure to hand back the corresponding watchboxes
                    if cam_id in ce_obj.config_watchboxes:
                        return_message = "watchboxes:" + str(ce_obj.config_watchboxes[cam_id])
                    else:
                        return_message = "watchboxes:[]"
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
                    time.sleep(10)
                    # Otherwise, this is data.
                    incoming_data = eval(decoded)
                    
                    
                    # Get the timestamp of this event:
                    frame_index_incoming = incoming_data[0]["time"]
                    data_to_write = [frame_index_incoming]
                    
                    
                    # ce_obj.result_output["incoming"].append([incoming_data, frame_index_incoming])
                    data_to_write.append(incoming_data)


                    ce_obj.update(incoming_data)
                    result, change_of_state, old_results = ce_obj.evaluate()

                    # ce_obj.result_output["events"].append([\
                    #                                       result, \
                    #                                       change_of_state, \
                    #                                       old_results, \
                    #                                       frame_index_incoming])
                    data_to_write.append([result, change_of_state, old_results])
                    
                    debug_file.write(":::".join([str(x) for x in data_to_write]) + "\n")
                    
                    # Check if any event became true
                    message_to_send = "none"
                    if result[3]:
                        event_turned_true = result[0][0]
                        message_to_send = event_turned_true
                    # Send this message back 
                    # sendMessage(message_to_send, addr, serverSocket)
                        

                    if change_of_state:
                        print()
                        print(old_results)
                        print(result)
                        
                        
                    
        except Exception as e:
            print(e)
            input()
    
    debug_file.close()


def build_ce1():
    # CE1
    # First, initialize our complex event
    ce1 = complexEvent()
    # Set up our watchboxes
    ce1.addWatchbox("bridgewatchbox0 = watchbox('camera3', positions=[1,1,1919,1079], id=0)")
    ce1.addWatchbox("bridgewatchbox1 = watchbox('camera3', positions=[213,274,772,772], id=1)")
    ce1.addWatchbox("bridgewatchbox2 = watchbox('camera3', positions=[816,366,1200,725], id=2)")
    ce1.addWatchbox("bridgewatchbox3 = watchbox('camera3', positions=[1294,290,1881,765], id=3)")

    # Now we set up the atomic events

    # First, four vehicles approach the bridge from one side
    ev11a = Event("bridgewatchbox0.composition(at=0, model='rec_vehicle').size==4 and bridgewatchbox0.composition(at=1, model='rec_vehicle').size!=4")

    # Next, we have two vehicles on either side of the bridge
    ev11b = Event( "bridgewatchbox1.composition(at=0, model='rec_vehicle').size==2 and bridgewatchbox3.composition(at=0, model='rec_vehicle').size==2" )

    # Then we have two vehicles exit watchbox 3
    ev11c1 = Event( "bridgewatchbox3.composition(at=1, model='rec_vehicle').size==2 and bridgewatchbox3.composition(at=0, model='rec_vehicle').size==0" )

    # And then two vehicles show up in watchbox 1
    # ev11c2 = Event( "bridgewatchbox1.composition(at=0, model='rec_vehicle').size==4 and bridgewatchbox1.composition(at=1, model='rec_vehicle').size!=4" )

    # And finally we add these events together
    ce1.addEventSequence([ ev11a, ev11b, GEN_PERMUTE(ev11c1, "size")])
    
    return ce1




import argparse


parser = argparse.ArgumentParser(description='Edge Processing Node')
parser.add_argument('--ce', type=int, help='Determines which CE we want to capture')
parser.add_argument('--server_port', type=int, help='Determines which CE we want to capture')
parser.add_argument('--result_dir', type=str, help='this is where we will output our json')
parser.add_argument('--debug_output', type=str, help='this is where we will output debug data')
args = parser.parse_args()
SERVER_ADDR = ("127.0.0.1", args.server_port)


if __name__=='__main__':
    
    complexEventObj = None
    if args.ce == 1:
        complexEventObj = build_ce1()
    
    complexEventObj.result_dir = args.result_dir
  
    # Our bounding boxes are as follows:
    watchboxes = {
        2: [ [1,1,1919,1079, 1], [213,274,772,772,1], [816,366,1200,725,1], [1294,290,1881,765,1]]
    }
    
    complexEventObj.config_watchboxes = watchboxes
    
    
    # Set up our server
    server_listen_thread = threading.Thread(target=server_listen, args=(complexEventObj,))
    server_listen_thread.start()

#     if incoming_data:
#         #### RUN OUR EVALUATION ON THE EVENT ANYTIME WE GET NEW DATA
#         ce1.update(incoming_data)

#         event_occurred, results = ce1.evaluate()
#         if event_occurred:

#             for result in results:
#                 print("Event %s has changed to %s at frame %d" %(result[0], str(result[1]), frame_index))

    
    