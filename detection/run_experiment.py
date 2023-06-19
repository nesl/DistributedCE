import os
import csv
import time
import subprocess
import psutil

from parse_utils import parse_gt_log, get_videos, get_takes_of_type
from LanguageCE.test_ce import build_ce1, build_ce2, build_ce3

import socket
from socket import SHUT_RDWR
import threading
import argparse
import traceback


def get_pid(name):
    return check_output(["pidof",name])


# Send data - encode it 
def sendMessage(message, addr, conn):
    # Turn the message into bytes
    message = str(message).encode()
    print("sending to " + str(addr))
    # sock.sendto(message, addr)
    conn.send(message)
        

# This is our listening server, which is where we get all of our results
#  This is the flow of information between server and client:
#  clients say hello - this establishes which clients are active
#  server sends video
#  Clients acknowledge they have received and loaded the video
#  Server sends start signal



def server_listen(ce_obj, ce_structure, server_addr, client_addr, video_path):

    # First, bind the server
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serverSocket.bind(server_addr)
    serverSocket.listen()
    
    # Open our file for writing
    # debug_filename = '/'.join([ce_obj.result_dir, "ce_output.txt"])
    # debug_file = open(debug_filename, "w", buffering=1)
    
    print("Listening on " + str(server_addr))

    # Send data to edge clients
    # time.sleep(2)

    # Say hello to clients
    # for client_addr in client_addresses:
    #     sendMessage("serverhello", client_addr, serverSocket)
    #     print("HERE")

    # Accept the connection from the client
    conn, addr = serverSocket.accept()
    print("Connection with " + str(addr) + " established")

    # Check to see what camera ID this is
    data = conn.recv(512)
    decoded = data.decode()
    cam_id = int(decoded.split(":")[1])
    ce_obj.client_info[cam_id] = addr
    print("Received camid of " + str(cam_id))

    print("Sending video data...")
    # Open the video file and transmit it
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
        # while video_data:
        conn.sendall(video_data)
        time.sleep(2)
        conn.send(b"done")
        # sendMessage("Hello", client_addr, serverSocket)
        # video_data = video_file.read()
    print("Sent video data!")


    # Now for the recv logic
    while True:
        
        data = conn.recv(512)
        if data:
            decoded = data.decode()

            # If this is a handshake message containing camera id
            # if "camera_id:" in decoded:
            #     cam_id = int(decoded.split(":")[1])
            #     # ce_obj.client_info[cam_id] = addr
            #     print(cam_id)
                

            # If this is a message on having received the video and is ready to run
            #  Send function information
            if "ready" in decoded:

                print(ce_obj.config_watchboxes)
                # Be sure to hand back the corresponding watchboxes
                if cam_id in ce_obj.config_watchboxes:
                    return_message = "watchboxes:" + str(ce_obj.config_watchboxes[cam_id])
                else:
                    return_message = "watchboxes:[]"
                print(return_message)
                sendMessage(return_message, addr, conn)


            elif "quitting:" in decoded:

                print("quitout!")

                # Write our data to file
                # result_file = '/'.join([ce_obj.result_dir, "ce_output.json"])
                # with open(result_file, "w") as wfile:
                #     json.dump(ce_obj.result_output, wfile)
                # break
                # debug_file.close()
                conn.shutdown(SHUT_RDWR)
                conn.close()
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
                        
                        
                    
        # except Exception as e:
        #     print(traceback.format_exc())
        #     input()
    
    # debug_file.close()

def setup_ce_detector(ce_index, server_addr, client_addresses, video_files):

    try:
        complexEventObj = None
        ce_structure = []
        if ce_index == 1:
            complexEventObj, ce_structure = build_ce1()
        
            # complexEventObj.result_dir = args.result_dir
        
            # Our bounding boxes are as follows:
            watchboxes = {
                2: [ [1,1,1919,1079, 1], [213,274,772,772,1], [816,366,1200,725,1], [1294,290,1881,765,1]],
                1: [ [413,274,1072,772, 0] ],
                0: [ [1,1,1919,1079, 1] ]
            }
            complexEventObj.config_watchboxes = watchboxes

        elif ce_index == 2:

            complexEventObj, ce_structure =  build_ce2()
            # complexEventObj.result_dir = args.result_dir

            watchboxes = {
                2: [ [1,1,1919,1079,1], [213,274,772,772,1], [816,366,1200,725,1], [1294,290,1881,765,1]],
                1: [ [1294,274,1881,765, 0] ],
                0: [ [1,1,1919,1079, 1] ]
            }
            complexEventObj.config_watchboxes = watchboxes

        elif ce_index == 3:

            complexEventObj, ce_structure =  build_ce3()
            # complexEventObj.result_dir = args.result_dir

            # Our bounding boxes are as follows:
            watchboxes = {
                1: [ [1,1,1919,1079, 0], [213,274,772,772,1], [750,366,1200,725,0], [1210,290,1750,765,1], [1,1,1919,1079,1] ],
                0: [ [1,1,1919,1079, 0], [1,1,1919,1079, 1] ]
            }
            complexEventObj.config_watchboxes = watchboxes

    except Exception as e:
        print(traceback.format_exc())
        input()
    
    

    # Set up some TCP threads for communication
    server_threads = []
    for c_i, client_addr in enumerate(client_addresses):
        # Set up our server
        server_listen_thread = threading.Thread(target=server_listen, \
            args=(complexEventObj,ce_structure,server_addr,client_addr,video_files[c_i],))
        server_listen_thread.start()
        server_threads.append(server_listen_thread)

    # Wait for all server threads to exit
    while True:
        time.sleep(1)
        # Check if all server threads are alive
        threads_status = [x.is_alive() for x in server_threads]
        if not any(threads_status):  # If all are dead, we can move on
            break

    print("FINISHED!")
    


    
# Get the process IDs of our camera and server programs
def get_program_pids(camera_program, server_program):
    
    camera_pid, server_pid = -1, -1
    
    ps = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE).communicate()[0]
    ps = ps.decode()
    processes = ps.split('\n')
    for x in processes:
        if camera_program in x:
            camera_pid = x.split()[1]
        elif server_program in x:
            server_pid = x.split()[1]
    
    return camera_pid, server_pid






def execute_main_experiment():

    server_port = 6792
    client_port = 6703

    server_addr = ("192.168.55.100", server_port)
    client_addr = ("192.168.55.1", client_port)
    client_addresses = [client_addr]
    video_files = [["../people.mp4"], ["../people.mp4"]]

    # Set up the server
    for videos_to_send in video_files:
        setup_ce_detector(1, server_addr, client_addresses, videos_to_send)



def execute_main_experiment2():

    server_port = 6792
    start_port = 6703

    # IMG_BUFFER_ZONE = 50
    # REMATCH_LOST_TRACKS = True
    

    # DOMAIN_SHIFT_TYPE = "none"  # Or alttank, or smoke
    
    # First thing we have to do:
    #  Figure out what CEs we want to run statistics over:
    ce_file = "../neuroplexLog.csv"
    # This is of the structure   { ce_number : [(take_number, [ae_name, ae_name, etc],[frame_index, frame_index, etc]), ....] }
    parsed_gt = parse_gt_log(ce_file)
    
    print("Found examples of the following events: ")
    print(parsed_gt.keys())
    
    # This is where we get all of our videos
    # video_parent_folder = "../videos"
    video_parent_folder = "/media/brianw/1511bdc1-b782-4302-9f3e-f6d90b91f857/home/brianw/SoartechData/videos"
    
    # video_parent_folder = "/media/tuscan-chicken/windows1/data"
    # events_of_interest = [("CE1", 'none', 0, False, False, 1),  ("NCE", 'none', 50, True, True, 1), \
    #                         ("CE2", 'none', 50, True, True, 2), ("ICE2", 'none', 50, True, True, 2), \
    #                         ("CE3", 'none', 50, True, True, 3), ("ICE3", 'none', 50, True, True, 3)]
    
    # Events of interest requires 5 parameters:
    #  The name of the CE (which will be used for naming folders AND looking up in the ground truth log)
    #  The type of domain shift, which can be "none", "alttank", or "smoke"
    #  The buffer around the image where we ignore detections (e.g. 10 means anything within 10px of the edge of the image has its detections ignored)
    #  Truth value for whether we rematch tracks that have been lost (e.g. try to re-assign a track ID when lost)
    #  Truth value for whether we ignore stationary detections (e.g. do not use YOLOv5 detections when objects are stationary)
    #  The number of complex event, which is used by the complex event server (e.g. 3 means it evaluates CE3)
    #  And whether or not we use ground truth annotations for detection/tracking.
    # events_of_interest = [("CE1", 'alttank', 50, True, True, 1), ("CE1", 'smoke', 50, True, True, 1)]
    events_of_interest = [("CE1", 'smoke', 50, True, True, 1, True), ("CE1", 'alttank', 50, True, True, 1, True)]
    # Check if we are using ground truth files - if so, only some will have them.

    
    
    event_count = {}
    print([(x,len(parsed_gt[x])) for x in parsed_gt.keys()])


    
    # Process names of our python programs
    camera_program = "python load.py"
    server_program = "python test_ce.py"
    # This is where we store our files
    result_folder = "ce_results"

    

    
    
    # Next, we need to get the events of interest and their takes
    for config in events_of_interest:

        ev = config[0]
        dshift_type = config[1]

        # Do we use an image buffer around the image for detections?
        IMG_BUFFER_ZONE = config[2]
        # Do we rematch any lost tracks?
        REMATCH_LOST_TRACKS = config[3]
        # Do we only weight detections lower when stationary?
        IGNORE_STATIONARY_DETECTIONS = config[4] 
        CURRENT_CE_EVALUATOR = config[5]  # Change this if you want to evaluate different CEs (e.g. 1 = CE1, 2 = CE2, etc)
        USING_GT = config[6]

        # Added string for our folder - basically the extra config for its name
        added_folder_names = "_".join([dshift_type, str(IMG_BUFFER_ZONE), str(REMATCH_LOST_TRACKS), str(IGNORE_STATIONARY_DETECTIONS), str(USING_GT)])

        
        # Add our current ce examples
        ev_folder = [result_folder, ev+"_"+added_folder_names]
        # Add folder if it doesn't exist
        if not os.path.exists('/'.join(ev_folder)):
            os.mkdir('/'.join(ev_folder))

        event_count[ev] = 0

        # Get the takes for this event
        relevant_takes = [x[0] for x in parsed_gt[ev]]
        # Get the takes for no domain shift
        relevant_takes = get_takes_of_type(video_parent_folder, relevant_takes, dshift_type, USING_GT)
        print(relevant_takes)
        print(len(relevant_takes))

        # Next, for every take, get the videos.
        for entry in relevant_takes:
            
            take = entry
            
            # We only care about take YYY:
            # if take != 345: # and ev == "ICE1" and dshift_type == "alttank":
            #     continue

            video_files, video_ids = get_videos(video_parent_folder, take)

            # Check if this particular take actually has videos - otherwise, skip.
            if not video_files:
                continue
            if len(video_files) > 3:  # Ok, temporarily skip folders with too many videos
                continue

            print("Current take :  " + str(take))

            # Update the number of examples for this event
            event_count[ev] += 1
            
            video_ids = [str(x) for x in video_ids]
            
            # Figure out where to store data for this take
            vtake_folder = [result_folder, ev+"_"+added_folder_names, str(take)]
            vtake_folder = '/'.join(vtake_folder)
            # Create the new folder if it does not exist
            if not os.path.exists(vtake_folder):
                os.mkdir(vtake_folder)
            # Set up the files that we will output to
            server_stdout_file = vtake_folder + "/server_debug.txt"
            camera_stdout_file = vtake_folder + "/camera_debug.txt"
            
            
            # We run a command for every set of videos
            command = []
            
            # Start the server for the current command
            server_command = "gnome-terminal -- bash ce_server.sh " + str(CURRENT_CE_EVALUATOR) + " " + str(server_port) + " " + vtake_folder + " " + server_stdout_file
            command.append(server_command)

            recover_lost_track = "--recover_lost_track" if REMATCH_LOST_TRACKS else "--no_recover_lost_track"
            ignore_stationary = "--ignore_stationary" if IGNORE_STATIONARY_DETECTIONS else "--no_ignore_stationary"
            use_gt = "--use_gt" if USING_GT else "--no_use_gt"
            
            # Start the script for each camera, sending data to the server
            # gnome-terminal -- 
            camera_command = "gnome-terminal -- bash camera.sh " + str(' '.join(video_files)) + " " +\
                    str(' '.join(video_ids) + " " + str(start_port) + " " + str(server_port)) + " " +\
                         vtake_folder + " " + str(take) + " " + recover_lost_track + " " + str(IMG_BUFFER_ZONE) + \
                         " " + str(ignore_stationary) + " " + str(use_gt)
            command.append(camera_command)
            
            # Run both the server and the camera item at the same time.
            command = ' & '.join(command)

            # Run the command 
            os.system(command)
            
            # Loop until our server process dies
            while True:
                #  Get the pids
                camera_pid, server_pid = get_program_pids(camera_program, server_program)
                
                # If the camera PID dies, then we close our server PID as well and break the loop
                if camera_pid == -1 and server_pid == -1:
                    # os.system("kill -9 " + str(server_pid))
                    break
                
                # We only need to check every second or so
                time.sleep(1)
            
            
                # Note - piping out of bash won't work for the server - it gets buffered until the program exits, but nothing is written when the script is killed.
                
    
                
                # First, we are looking for three python running processes.
                #  One is this program, another is the camera program, and lastly the server.
                #   The camera program will always exit when it is completed, so we wait 
                
            # Importantly, you must save their output to a file, in addition to doing so within the file itself (former is for error checking, latter is for actual results).
            
            
            # Now, be sure to wait until both processes either die (killed by user) or ended
            #  properly (only load.py will 'end' - on the other hand, the server test_ce will keep listening)
            
        # End for loop that iterates over each take
        


    # Show how many events we are counting
    print(event_count)


if __name__ == "__main__":

    execute_main_experiment()
    
            

            






