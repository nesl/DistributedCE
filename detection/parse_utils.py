import csv
import os

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
    takes = []
    with open(file, "r") as rfile:
        gt_reader = csv.reader(rfile, delimiter=',')
        
        current_take = -1
        current_ce_number = ""
        current_ae = []
        current_frame_indexes = []
        # Iterate through every row of the CSV
        for row in gt_reader:

            # Stop the reader if the next row is blank
            if not row:
                break

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

                    takes.append(take_number)
                
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
            is_video = False
            if "cam1" in video_file:
                video_ids.append(0)
                is_video = True
            elif "cam2" in video_file:
                video_ids.append(1)
                is_video = True
            elif "cam3" in video_file:
                video_ids.append(2)
                is_video = True

            if is_video:
                video_files.append(take_folder + "/" + video_file)
    
    return video_files, video_ids


# Get takes where a domain shift happens based on keyword
def get_takes_of_type(video_parent_folder, takes, keyword):

    relevant_takes = []
    for take in takes:
        # Get all the video files
        vfiles, v_ids = get_videos(video_parent_folder, take)

        # Depending on our keyword, we have different checks
        if keyword == "alttank":
            if any([keyword in x for x in vfiles]):
                relevant_takes.append(take)
        elif keyword == "smoke":
            if any([keyword in x for x in vfiles]):
                relevant_takes.append(take)
        elif keyword == "snow":
            if any([keyword in x for x in vfiles]):
                relevant_takes.append(take)
        else:
            to_check = "take_"+str(take)+".mp4"
            if any([to_check in x for x in vfiles]):
                relevant_takes.append(take)
            # relevant_takes.append(take)

    return relevant_takes



if __name__ == "__main__":

    ce_file = "../neuroplexLog.csv"
    # This is of the structure   { ce_number : [(take_number, [ae_name, ae_name, etc],[frame_index, frame_index, etc]), ....] }
    
    # video_parent_folder = "/media/tuscan-chicken/3CC47BFDC47BB7AA/Users/Brian/Downloads/data"
    video_parent_folder = "/media/brianw/1511bdc1-b782-4302-9f3e-f6d90b91f857/home/brianw/SoartechData/videos/"
    complex_event = "CE3"
    domain_shift = "none"

    parsed_gt = parse_gt_log(ce_file)

    # Total number of parsed items
    print(sum([len(x) for x in parsed_gt.values()]))

    # Show all takes involving CE1
    relevant_takes = [x[0] for x in parsed_gt[complex_event]]

    # Show all takes involving some domain shift or none
    relevant_takes = get_takes_of_type(video_parent_folder, relevant_takes, domain_shift)

    print(relevant_takes)
    print(len(relevant_takes))
    

        
