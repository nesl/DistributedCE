import os
import csv
from statistics import median, mean
from parse_utils import parse_gt_log, get_takes_of_type

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


# times is of type [frame index of ground truth, frame index of detection]
#  If the time difference is negative, it means the detected frame occurred after
def time_difference(time0, time1, fps=30):

    time_diff = -1

    # Get frame difference 
    frame_diff = time0 - time1
    # Convert to seconds
    time_diff = -frame_diff / fps

    return time_diff
    

# Get mean and median time in differences for each AE
# time_diffs_by_ae is cam_folder : { event : time difference }
def get_ae_time_errors(matches):

    time_errors_by_ae = {}  # is like event_name : [time_diff, time_diff, etc]
    
    # Iterate through cam_folder and then each event there
    for cam_folder in matches.keys():
        for ev in matches[cam_folder]:

            if ev not in time_errors_by_ae:
                time_errors_by_ae[ev] = []

            # If the event is detected, get the time difference
            time_errors_by_ae[ev].append(matches[cam_folder][ev])
    
    # Now we calculate the median and mean of each atomic event
    median_time_errors = {}
    mean_time_errors = {}

    for ae_name in time_errors_by_ae.keys():
        median_time_errors[ae_name] = median(time_errors_by_ae[ae_name])
        mean_time_errors[ae_name] = mean(time_errors_by_ae[ae_name])

    return median_time_errors, mean_time_errors


# Get the accuracy for all events
def get_accuracy(matches, gt_events):

     # Check the time diffs for each type of AE
    accuracy_by_ae = {}
    accuracy_by_ce = []

    # Iterate through every ground truth event for every cam folder
    for cam_folder in gt_events.keys():

        # accuracy_by_ce_per_cam[cam_folder] = 0

        num_matches = 0
        for gt_ev in gt_events[cam_folder]:

            if gt_ev not in accuracy_by_ae.keys():
                accuracy_by_ae[gt_ev] = []

            # Check if the matched event captured this event
            if gt_ev in matches[cam_folder].keys():

                # Get time diff
                detected_time_diff = matches[cam_folder][gt_ev]
                
                # And the accuracy
                accuracy_by_ae[gt_ev].append(1.0)
                # Update the count for the number of matches
                num_matches += 1
            else:
                accuracy_by_ae[gt_ev].append(0.0)

        # Update the completion accuracy
        # accuracy_by_ce_per_cam[cam_folder] = num_matches / len(gt_events[cam_folder])

        # Check if our ce was completed (all events matched)
        if num_matches == len(gt_events[cam_folder]):
            accuracy_by_ce.append(1.0)
        else:
            accuracy_by_ce.append(0.0)
    # Now count up our accuracy by ce
    accuracy_by_ce = mean(accuracy_by_ce)
    # Now also get the accuracy by ae
    for ae in accuracy_by_ae.keys():
        accuracy_by_ae[ae] = mean(accuracy_by_ae[ae])


    return accuracy_by_ae, accuracy_by_ce

# Get the number of false alarms by ae
#  false_alarms made up of cam_folder : [event_name, ... ]
def get_false_alarms(false_alarms):

    false_alarms_by_ae = {}
    false_alarms_by_ce = 0

    # Iterate through each cam folder
    for cam_folder in false_alarms.keys():

        for ev in false_alarms[cam_folder]:
            if ev not in false_alarms_by_ae:
                false_alarms_by_ae[ev] = 0
            false_alarms_by_ae[ev] += 1
        
        if false_alarms[cam_folder]:
            false_alarms_by_ce += 1
    
    return false_alarms_by_ae, false_alarms_by_ce
        
# Get the number of missed detections by ae
# Made up of cam_folder : [event_name, ...]
def get_missed_detections(missed_detections):

    missed_detections_by_ae = {}
    missed_detections_by_ce = 0

    # Iterate through each cam folder
    for cam_folder in missed_detections.keys():
        # Iterate through every missed event
        for ev in missed_detections[cam_folder]:

            # If we haven't seen this event before, initialize it
            if ev not in missed_detections_by_ae:
                missed_detections_by_ae[ev] = 0
            
            missed_detections_by_ae[ev] += 1
        
        if missed_detections[cam_folder]:
            missed_detections_by_ce += 1
    
    
    
    return missed_detections_by_ae, missed_detections_by_ce


if __name__ == "__main__":

    # CE1 mapping - this is how we compare different event names
    ce1_mapping = {
        "1.1z": "vehicles_head_to_bridge", \
        "1.1a": "vehicles_approach_bridge", \
        "1.1b": "tanks_present_vehicles_plant_bombs", \
        "1.1c": "vehicles_leave" \
    }

    # CE2 mapping
    ce2_mapping = {
        "2.1z": "vehicle_move", \
        "2.1a": "vec_approach_bridge", \
        "2.1b": "rec_on_either_side", \
        "2.3": "tanks_exit_bridge", \
        "CE2": "tank_first_rec_first"
    }

    # CE3 mapping
    ce3_mapping = {
        "3.0a": "all_in_cam1_all_in_cam0", \
        "3.1": "recce_in_opposite_wb", \
        "3.2": "tank_in_middle"
    }

    #### START IMPORTANT CONFIG VARIABLES #####
    ce_mapping = ce3_mapping
    ce_of_interest = "CE3"
    DOMAIN_SHIFT_TYPE = "none"  # Can be "none", "smoke", or "alttank"
    ce_result_folder = "CE3_none_50_True_True"
    # result_folder = "/media/brianw/1511bdc1-b782-4302-9f3e-f6d90b91f857/home/brianw/SoartechData/ce_results/" + ce_result_folder
    # result_folder = "/media/brianw/Samsung USB/ce_results/" + ce_result_folder
    # result_folder = "/media/brianw/Elements/code/ce_results/" + ce_result_folder
    result_folder = "ce_results/" + ce_result_folder
    ce_file = "../neuroplexLog.csv"
    video_parent_folder = "/media/brianw/1511bdc1-b782-4302-9f3e-f6d90b91f857/home/brianw/SoartechData/videos"
    #### END IMPORTANT CONFIG VARIABLES #####

    # This is of the structure   { ce_number : [(take_number, [ae_name, ae_name, etc],[frame_index, frame_index, etc]), ....] }
    parsed_gt = parse_gt_log(ce_file)
    camera_folders = os.listdir(result_folder)
    
    gt_event_logs = parsed_gt[ce_of_interest]

    # Get only the takes relevant to use
    relevant_takes = [x[0] for x in parsed_gt[ce_of_interest]]
    relevant_takes = get_takes_of_type(video_parent_folder, relevant_takes, DOMAIN_SHIFT_TYPE)
    print(relevant_takes)
    chosen_metric = binary_metric
    
    false_alarms = {}  # made up of cam_folder : [event_name, ... ]
    missed_detections = {}  # Made up of cam_folder : [event_name, ...]
    matches = {}  # Made up cam_folder : { event_name : metric, ... }
    gt_events = {}  # Made up cam_folder :  [event_name, ...]

    # Keep track of the count of each event for both gt and detected
    detected_event_count = {} # made up of event_name : count
    gt_event_count = {} # Made up of event_name : count

    # Iterate through each camera folder, getting its ce_output
    #  camera_folder is just the folder for our results, it's not the actual video folder
    for cam_folder in camera_folders:

        if int(cam_folder) not in relevant_takes:  # Skip if not relevant
            continue

        print("Parsing for %s" %(cam_folder))

        # Get the ce file
        ce_file = '/'.join([result_folder, cam_folder, "ce_output.txt"])


        # Open the ce file and parse it it
        ce_result_gt = {} # Entries of event name : [timestamp]
        ce_result_det = {}  # Same as above
    
        # Next, iterate through the gt_event_logs to fill the results
        #  For this particular take
        for tup in gt_event_logs:
            if tup[0] == int(cam_folder):  # Match the tuple to the current CE result
                for ev_i, ev in enumerate(tup[1]):
                    if ev in ce_mapping.keys():
                        translated_event = ce_mapping[ev]
                        ev_frame_index = tup[2][ev_i]
                        ce_result_gt[translated_event] = ev_frame_index
                        

        # Parse ground truth event log
        previous_event_time_diffs = {} # Entries of event_name : min_frame_diff
        with open(ce_file, "r") as rfile:
            # Iterate through every row of the CSV
            for row in rfile:
                # Split by :::
                frame_index = int(row.split(":::")[0])
                event = eval(row.split(":::")[2])
                
                if event[0]:

                    event_name = event[0][0]
                    event_name = '_'.join(event_name)
                    event_happened = event[0][3]

                    # Check if the event actually happened.
                    #  And check if any events are missing
                    if event_happened:
                        
                        # Note - if this already exists in our GT
                        #  make sure to get the closest version of this event
                        #  (in cases where the event occurs again)
                        if event_name in ce_result_gt:
                            current_diff = abs(ce_result_gt[event_name] - frame_index)
                            if event_name not in previous_event_time_diffs:
                                previous_event_time_diffs[event_name] = current_diff
                                ce_result_det[event_name] = frame_index
                            elif previous_event_time_diffs[event_name] > current_diff:
                                previous_event_time_diffs[event_name] = current_diff
                                ce_result_det[event_name] = frame_index
                        else:
                            ce_result_det[event_name] = frame_index
                        
                        


        false_alarms[cam_folder] = []
        missed_detections[cam_folder] = []
        gt_events[cam_folder] = []
        matches[cam_folder] = {}
        # Iterate through each ce result for both det and gt
        for detected_ev in ce_result_det.keys():
            # The atomic event is detected but not in the GT log, so we have a false alarm
            if detected_ev not in ce_result_gt.keys():
                false_alarms[cam_folder].append(detected_ev)
            else:  # We have a match - record the time differences here.
                matches[cam_folder][detected_ev] = time_difference(ce_result_gt[detected_ev], \
                    ce_result_det[detected_ev])

            if detected_ev not in detected_event_count:
                detected_event_count[detected_ev] = 0
            detected_event_count[detected_ev] += 1
                

        for gt_ev in ce_result_gt.keys():
            # If the atomic event is not detected but is in the GT log, we have a miss
            if gt_ev not in ce_result_det.keys():
                missed_detections[cam_folder].append(gt_ev)
            gt_events[cam_folder].append(gt_ev)


    for x in sorted(missed_detections.keys()):
        print("key: " + str(x) + " " + str(missed_detections[x]))

    # Measure overall accuracy, both at the CE level and AE level
    median_time_errors_by_ae, mean_time_errors_by_ae = get_ae_time_errors(matches)
    accuracy_by_ae, accuracy_by_ce = get_accuracy(matches, gt_events)
    false_alarms_by_ae, false_alarms_by_ce = get_false_alarms(false_alarms)
    missed_detections_by_ae, missed_detections_by_ce = get_missed_detections(missed_detections)


    # Print completion percentage
    print("Completion Percentage: %f " % (accuracy_by_ce))
    print("False alarms at CE level: %d / %d " % (false_alarms_by_ce, len(matches.keys())))
    print("Missed detections at CE level: %d / %d " % (missed_detections_by_ce, len(matches.keys())))

    # Now iterate through each atomic event
    for ae in accuracy_by_ae.keys():

        # Print our results
        print("Measuring for AE: %s" % (ae))
        print("\tAccuracy: %f " % (accuracy_by_ae[ae]))
        print("\tMedian error in start time: %f seconds" % (median_time_errors_by_ae[ae]))
        print("\tMean error in start time: %f seconds" % (mean_time_errors_by_ae[ae]))

        # Print the false alarms
        if ae in false_alarms_by_ae:
            print("\tNumber of false alarms (out of detections): %d " % (false_alarms_by_ae[ae]))
        if ae in missed_detections_by_ae:
            print("\tNumber of missed detections (out of gt): %d " % (missed_detections_by_ae[ae]))


    print("\n\nFalse Alarm Distribution: ")
    # Iterate through all false alarms, and count out of how many there were          
    for ae in false_alarms_by_ae.keys():
        print("Measuring for AE: %s" % (ae))
        print("\tNumber of false alarms (out of detections): %d / %d" % (false_alarms_by_ae[ae], detected_event_count[ae]))
            
        
                        
        
                    