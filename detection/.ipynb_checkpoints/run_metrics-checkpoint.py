import os
import csv

from run_experiment import parse_gt_log

# So here's the main way of measuring accuracy:
#  - did the event happen, and did it happen accurately?
#     The latter part is difficult to measure - we could say 
#     the prediction is accurate if it is within a threshold, like 30 frames? Basically binary.
#     Or, we can say that an event can occur within 300 frames (~10seconds), with linearly decreasing
#      accuracy.  This might be a good option.


# Simple accuracy metric - occurs within x frames == 100% accurate (binary)
def binary_metric(times, bound=900): # By default, 300 frames is the limit
    
    accuracy = 1.0
    if len(times) < 2:
        accuracy = 0.0
    elif abs(times[1] - times[0]) > bound:
        accuracy = 0.0
        
    return accuracy

# Another metric - distance based accuracy
def distance_metric(times, bound=300): # by default 300 frames is the limit
    
    accuracy = 0.0
    if len(times) < 2:
        accuracy = 0.0
    elif abs(times[1] - times[0]) > bound:
        accuracy = 0.0
    else:
        accuracy = 1.0 - (abs(times[1] - times[0]) / 300)
    
    return accuracy
    

if __name__ == "__main__":
    
    
    # CE1 mapping - this is how we compare different event names
    ce1_mapping = {
        "1.1a": "ev11a", \
        "1.1b": "ev11b", \
        "1.1c": "ev11c1" \
    }
    
    
    # List out our result folder
    ce_of_interest = "CE1"
    result_folder = "ce_results/" + ce_of_interest
    camera_folders = os.listdir(result_folder)
    
    ce_file = "../neuroplexLog.csv"
    # This is of the structure   { ce_number : [(take_number, [ae_name, ae_name, etc],[frame_index, frame_index, etc]), ....] }
    parsed_gt = parse_gt_log(ce_file)
    
    gt_event_logs = parsed_gt[ce_of_interest]
    
    chosen_metric = binary_metric
    
    results = {}
    separate_results = {}
    # Iterate through each camera folder, getting its ce_output
    for cam_folder in camera_folders:
        
        # Get the ce file
        ce_file = '/'.join([result_folder, cam_folder, "ce_output.txt"])
        # Open the ce file and parse it it
        ce_result = {} # Entries of event name : [timestamp]
    
        # Next, iterate through the gt_event_logs to fill the results
        for tup in gt_event_logs:
            if tup[0] == int(cam_folder):
                for ev_i, ev in enumerate(tup[1]):
                    if ev in ce1_mapping.keys():
                        translated_event = ce1_mapping[ev]
                        ev_frame_index = tup[2][ev_i]
                        ce_result[translated_event] = [ev_frame_index]

        with open(ce_file, "r") as rfile:
            # Iterate through every row of the CSV
            for row in rfile:
                # Split by :::
                frame_index = int(row.split(":::")[0])
                event = eval(row.split(":::")[2])
                event_name = event[0][0][0]
                event_happened = event[0][3]
                if event_happened:
                    ce_result[event_name].append(frame_index)
                    
        # Now, go through each event and see how accurate it is
        temp_result = {}
        for ev in ce_result.keys():
            if ev not in results:
                results[ev] = [binary_metric(ce_result[ev])]
            else:
                results[ev].append(chosen_metric(ce_result[ev]))
            temp_result[ev] = chosen_metric(ce_result[ev])
        separate_results[cam_folder] = temp_result
        
    # We have to do some processing on the results - taking the average
    for rkey in results.keys():
        results[rkey] = sum(results[rkey])/len(results[rkey])
    print(results)
    for rkey in sorted(separate_results.keys()):
        print(rkey, separate_results[rkey])
            
            
        
            
        
                        
        
                    