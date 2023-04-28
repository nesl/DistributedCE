import os
import csv
import time
import subprocess
import psutil

from parse_utils import parse_gt_log, get_videos, get_takes_of_type


def get_pid(name):
    return check_output(["pidof",name])


                

    
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
    events_of_interest = [("CE3", 'none', 50, True, True, 3)]
    
    
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

        # Added string for our folder - basically the extra config for its name
        added_folder_names = "_".join([dshift_type, str(IMG_BUFFER_ZONE), str(REMATCH_LOST_TRACKS), str(IGNORE_STATIONARY_DETECTIONS)])

        
        # Add our current ce examples
        ev_folder = [result_folder, ev+"_"+added_folder_names]
        # Add folder if it doesn't exist
        if not os.path.exists('/'.join(ev_folder)):
            os.mkdir('/'.join(ev_folder))

        event_count[ev] = 0

        # Get the takes for this event
        relevant_takes = [x[0] for x in parsed_gt[ev]]
        # Get the takes for no domain shift
        relevant_takes = get_takes_of_type(video_parent_folder, relevant_takes, dshift_type)
        print(relevant_takes)
        print(len(relevant_takes))

        # Next, for every take, get the videos.
        for entry in relevant_takes:
            
            take = entry
            
            # We only care about take YYY:
            if take not in [299, 301, 324, 334, 337]:
                continue
            # if take != 181: # and ev == "ICE1" and dshift_type == "alttank":
            #     continue

            # Ignore existing takes
            # if str(take) in os.listdir('/'.join(ev_folder)):
            #     continue
            # if ev == "CE1" and int(take) not in [75, 124, 142, 149, 150, 168, 174, 199, 344]:
            #     continue
            # if ev == "NCE" and int(take) not in [78, 136, 137, 138, 146, 147, 148, 153, 154, 155, 159, 160, 161, 165, 166, 167, 171, 172, 173, 177, 178, 179, 190, 191, 192, 196, 197, 198]:
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

            
            # Start the script for each camera, sending data to the server
            # gnome-terminal -- 
            camera_command = "gnome-terminal -- bash camera.sh " + str(' '.join(video_files)) + " " +\
                    str(' '.join(video_ids) + " " + str(start_port) + " " + str(server_port)) + " " +\
                         vtake_folder + " " + str(take) + " " + recover_lost_track + " " + str(IMG_BUFFER_ZONE) + \
                         " " + str(ignore_stationary)
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
    
            

            






