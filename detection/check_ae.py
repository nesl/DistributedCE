import os
import subprocess
import numpy as np
from tqdm import tqdm

def parse_output(output):
    # Get the following fields:
    # Split up the data
    output = output.decode()
    
    data_split = output.split(":::")
    frame_index = data_split[0]
    detections = data_split[1]
    trackers = data_split[2]
    
    trackers = trackers.replace("array", "")
    trackers = eval(trackers)
    # Get tracking information
    print(detections)
    for tkey in trackers.keys():

        # Get only the small list of data
        truncated_data = trackers[tkey][0:3]
        print(tkey, truncated_data)


# Get the file
video = "75"
file_of_interest = "ce_results/CE1/"+video+"/ae0.txt"

frame_range = [2781, 3000]
frame_index = [x for x in range(frame_range[0], frame_range[1], 3)]

for fi in tqdm(frame_index):
    
    # Grep for this frame
    command = "cat " + file_of_interest + " | grep " + str(fi) + ":::"
    output = subprocess.check_output(command, shell=True)
    
    # Now, parse the output
    parse_output(output)
    print("\n\n")