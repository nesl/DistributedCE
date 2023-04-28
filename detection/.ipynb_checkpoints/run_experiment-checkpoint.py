import os
import csv
import time
import subprocess
import psutil


def get_pid(name):
    return check_output(["pidof",name])

# Converts a time in seconds to frame index
def convert_time_to_frame(event_time, fps=30):
    
    frame_index = int(event_time * fps)
    return frame_index


# Parse a row of the CSV file
#  We are only interested in a few rows - so filter them here.
def parse_gt_row(row):
    frame_index = convert_time_to_frame(float(row[1]))
    take_number = int(row[2].split("take")[1].strip())
    event_name = row[3]
    
    return take_number, frame_index, event_name


# Check which event we are POTENTIALLY looking at:
def get_event_title(event_name):
    
    event_title = ""
    if "CE1DestroyTarget" in event_name:
        event_title = "CE1"
    elif "CE2MoveArmorAcrossTarget" in event_name:
        event_title = "CE2"
    elif "CE3DefensivePosition" in event_name:
        event_title = "CE3"
    else:
        event_title = "NCE"
        
    return event_title


# Here we parse the ground truth file into a specific data structure
# We need to get the following data structure:
#  { ce_number : [(take_number, [ae_name, ae_name, etc],[frame_index, frame_index, etc]), ....] }
#  There are several possible CE numbers:
#   CE1, CE2, CE3, NCE, ICE1, ICE2, etc - the ICE means incomplete CEs, and NCE means non-CEs.
def parse_gt_log(file):

    # Read in our file
    parsed_structure = {"CE1":[], "CE2":[], "CE3":[], "NCE":[], "ICE1":[], "ICE2":[], "ICE3":[]}
    with open(file, "r") as rfile:
        gt_reader = csv.reader(rfile, delimiter=',')
        
        current_take = -1
        current_ce_number = ""
        current_ae = []
        current_frame_indexes = []
        # Iterate through every row of the CSV
        for row in gt_reader:
            # If this is the first row for this take, then we just track the title, e.g. "Start Scenario CE3DefensivePosition"
            take_number, frame_index, event_name = parse_gt_row(row)
            
            if take_number != current_take:
                
                # First, check if we need to append our existing data
                if current_take != -1:
                    
                    # Note - sometimes we can only figure out if this is an ICE at the end of the entry
                    #  We can basically match the end ae with the ce title to see if this is necessary.
                    if current_ce_number != "NCE" and current_ae[-1] != current_ce_number:
                        current_ce_number = "I"+current_ce_number
                    
                    # Create our entry
                    entry = (current_take, current_ae, current_frame_indexes)
                    parsed_structure[current_ce_number].append(entry)
                
                # Once previous data has been appended, start a new entry
                current_ce_number = get_event_title(event_name)
                # Update the current take number
                current_take = take_number
                # Clear previous data
                current_ae = []
                current_frame_indexes = []
                
            else:  # Now we just add data to our entries
                
                # First, there's an annoying bug where the same take will show up for different events.  Just ignore previous cases.
                if frame_index == 0: # We have to re-update the current ce number and whatnot
                    current_ce_number = get_event_title(event_name)
                    current_take = take_number
                    current_ae = []
                    current_frame_indexes = []
                else:
                    current_ae.append(event_name.strip())
                    current_frame_indexes.append(frame_index)
                
    # No more parsing, just return the results
    return parsed_structure
                
# Get the corresponding videos, with their corresponding camera id:
def get_videos(parent_folder, take):
    
    take_folder = '/'.join([parent_folder, "take_"+str(take)])
    video_files = []
    video_ids = []
    
    # Make sure take folder actually exists
    if os.path.exists(take_folder):
        # Get the files
        for video_file in os.listdir(take_folder):

            if "cam1" in video_file:
                video_ids.append(0)
            elif "cam2" in video_file:
                video_ids.append(1)
            elif "cam3" in video_file:
                video_ids.append(2)
            video_files.append(take_folder + "/" + video_file)
    
    return video_files, video_ids
    
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

if __name__ == "__main__":

    
    server_port = 6792
    start_port = 6703
    
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
    events_of_interest = ["CE1"]
    current_event = 1
    
    print([(x,len(parsed_gt[x])) for x in parsed_gt.keys()])
    
    # Process names of our python programs
    camera_program = "python load.py"
    server_program = "python test_ce.py"
    # This is where we store our files
    result_folder = "ce_results"
    
    # Next, we need to get the events of interest and their takes
    for ev in events_of_interest:
        
        # Add our current ce examples
        ev_folder = [result_folder, ev]
        # Add folder if it doesn't exist
        if not os.path.exists('/'.join(ev_folder)):
            os.mkdir('/'.join(ev_folder))
        
        # Next, for every take, get the videos.
        for entry in parsed_gt[ev]:
            
            take = entry[0]
            
            video_files, video_ids = get_videos(video_parent_folder, take)

            # Check if this particular take actually has videos - otherwise, skip.
            if not video_files:
                continue
            if len(video_files) > 3:  # Ok, temporarily skip folders with too many videos
                continue
            
            video_ids = [str(x) for x in video_ids]
            
            # Figure out where to store data for this take
            vtake_folder = [result_folder, ev, str(take)]
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
            server_command = "gnome-terminal -- bash ce_server.sh " + str(current_event) + " " + str(server_port) + " " + vtake_folder + " " + server_stdout_file
            command.append(server_command)
            
            # Start the script for each camera, sending data to the server
            # gnome-terminal -- 
            camera_command = "gnome-terminal -- bash camera.sh " + str(' '.join(video_files)) + " " + str(' '.join(video_ids) + " " + str(start_port) + " " + str(server_port)) + " " + vtake_folder + " " + str(take)
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
            
            

            






