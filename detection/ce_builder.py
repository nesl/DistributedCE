#  This is our file for handling various objects and classes

import re
import traceback
import inspect
from transitions import Machine
from transitions.extensions import HierarchicalMachine
from transitions.extensions import GraphMachine
from transitions.extensions.factory import HierarchicalGraphMachine
import json
import itertools

# from antlr4 import *
# from antlr4.tree.Trees import Trees
# from antlr.languageLexer import languageLexer  # This is the lexer 
# from antlr.language import language   # This is the parser


import os, sys, inspect, io
from IPython.display import Image, display, display_png

import cv2
import numpy as np



# Always recompile the ANTLR language
# os.system("bash run_test.sh")


def get_param_name(var):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-3]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    return vars_name

#  So, we have groups, objects, locations/watchboxes.
#   Each of them have functions/relations with other entities
#        which generate spatial events.
class sensor_event_stream:

    # Set up a video stream.
    def __init__(self, id):
        self.id = id


class watchboxResult:
    
    def __init__(self, size=0, speed=0, color='red'):
        self.size = size
        self.speed = speed
        self.color = color
        
class watchbox:

    # Set up a watchbox
    def __init__(self, camera_id, positions, watchbox_id):
        self.camera_id = int(camera_id)
        self.positions = positions
        self.watchbox_id = watchbox_id
        self.data = [watchboxResult()]
        self.max_history = 1000
        
    # update the max history
    def updateMaxHistory(self, history_len):
        if self.max_history < history_len:
            self.max_history = history_len
        
    # Returns composition at a particular time
    #  This looks backwards in data
    def composition(self, at, model):

        if self.data:
            # Get the data at a timestamp
            #  0 is -1, 1 is -2, etc.
            
            # for x in self.data:
            #     print(x.size)

            composition_result = self.data[-at - 1]
            # print("data for cam " + str(self.camera_id) + " and id " + str(self.watchbox_id))
            # print(composition_result)
            return composition_result
        
    

    # Updates some internal values
    def update(self, data):
        
        # Remember - we are updating a list of dictionaries
        
        
        # First off, how many vehicles are there already?
        previous_count = self.data[-1].size
        new_count = previous_count

        # Get the camera id
        incoming_cam_id = data[0]["camera_id"]


        # And does this update take away, or add a vehicle?
        results = data[0]["results"]


        # Get the relevant matching watchbox
        for c_i, current_wb_id in enumerate(results[0][0]):
            # Make sure the wb id matches and the camera id matches
            if self.watchbox_id == current_wb_id and self.camera_id == int(incoming_cam_id): # watchbox id matches and same with the camera id

                print("Updating for watchbox ID" + str(self.watchbox_id) + " and cam " + str(self.camera_id))
                
                # print(self.data[-1])
                # If we add or subtract
                if results[0][1][c_i]:
                    new_count += 1
                else:
                    new_count -= 1
            
                # Append to our data
                self.data.append(watchboxResult(size=new_count))
                print("Current watchbox count is " + str(new_count))
                # Also make sure our max_history_length is being enforced
                while len(self.data) > self.max_history+1: # +1 because current timestamp is included
                    self.data.pop(0)


# So our CEs will need a few things:
#  First is obviously enumeration of states, such that they can be hierarchical
#   Of course, then we have the initial state.
#  Then we add our transitions, which can run functions before/after the transition.
#    Each transition needs a function trigger...

class Model:
    def clear_state(self, deep=False, force=False):
        print("Clearing state ...")
        return True
    
class OR:
    def __init__(self, *args):
        # Basically needs to return the full string but and/or
        # output = ' or '.join(["("+x.event+")" for x in args])
        
        output = [x.event for x in args]
        

        output.append("or")
        
        self.event = output

            
class AND:
    def __init__(self, *args):
        # Basically needs to return the full string but and/or
        # output = ' and '.join(["("+x.event+")" for x in args])
        output = [x.event[0] for x in args]
        output.append("and")
        self.event = output
        



class WITHIN:
    # Initializing a WITHIN statement is a bit tricky, as it refers to checking when
    #  an event becomes true within a certain amount of time.
    #  So it also requires an additional statement that checks if one time from one event is within another
    def __init__(self, *args):
        
        # First, get the time parameter and ignore the rest
        time_param = args[-1]
        statements = args[:-1]
        
        # generate the statement to execute
        output = [x.event[0] for x in statements]
        output.append(["within", None, time_param])

        self.event = output

        
        
# So how should this function work?
#  Firstly, it must be able to expand the maximum history -
#   e.g. at=1 size==2 and at=0 size==0
#       should have another option: at=2 size==2 and at=1 size==1 and at=0 size==0
#  So it's really a matter of 'filling out the middle'.
class GEN_PERMUTE:
    
    def get_attributes(self, statement, attribute):
        
        # First, get the time
        time = statement.split("at=")[1].split(",")[0]
        # Then, get the attribute value
        attr_value = statement.split(attribute)[1][2:]
        
        return (int(time), int(attr_value))
    
    # Now, we must generate permutations
    #  We must generate all possible permutations between these values
    def gen_combinations(self, attribute_values, times):
        
        # First, we must note the order: Is time increasing?  Are the attribute values increasing?
        time_ascending = times[0] < times[1]
        min_time = min(times)
        attributes_ascending = attribute_values[0] < attribute_values[1]
        
        # Next, figure out all subsets of those attribute values
        smallest_attribute_value = min(attribute_values)
        largest_attribute_value = max(attribute_values)
        # First, get all possible values (e.g. treat attribute_values as a range)
        new_possible_values = [i for i in range(smallest_attribute_value, largest_attribute_value+1)]
        
        possible_combinations = []
        # Now, get all possible combos which contain both values
        for L in range(len(new_possible_values) + 1):
            for subset in itertools.combinations(new_possible_values, L):
                # Make sure we have both values in it
                if smallest_attribute_value in subset and largest_attribute_value in subset:
                    possible_combinations.append(list(subset))
        
        # Now, we have to make sure our times and our attributes follow the same order
        #  as when we had read them in
        results = []
        for combo in possible_combinations:
            # Generate the time in order
            current_times = [x for x in range(min_time, min_time+len(combo))]
            if not time_ascending:
                current_times.sort(reverse=True)
            
            # Generate the combinations in order
            combo.sort()
            if not attributes_ascending:
                combo.sort(reverse=True)
                
            results.append((current_times, combo))
        
        return results
        
    # Note - combos is like ([list of times],[list of attribute values])
    def generate_statement_from_combo(self, combo, attribute, statement):
        
        # Get the text of the time that we should replace
        original_time_statement = "at="+statement.split("at=")[1].split(",")[0]
        original_attribute_statement = attribute + statement.split(attribute)[1]
        
        # Now, get the starting statements
        starting_statement_time = "at="
        starting_statement_attr = ''.join([x for x in original_attribute_statement if not x.isdigit()])
        
        # And now replace with the combo values
        times = combo[0]
        att_values = combo[1]
        completed_statement = []
        for i in range(len(times)):
            replacing_time_statement = starting_statement_time + str(times[i])
            replacing_att_statement = starting_statement_attr + str(att_values[i])
            
            # Here we replace our values with the new ones
            modified_statement = statement.replace(original_time_statement, replacing_time_statement)
            modified_statement = modified_statement.replace(original_attribute_statement, \
                                                                replacing_att_statement)
            completed_statement.append(modified_statement)
            
        completed_statement = " and ".join(completed_statement)
        
        return completed_statement
        
    
    def __init__(self, event, attribute):
        
        # First, we should determine what the possible transitions are between 
        #  the values involved with the attributes
        split_statements = event.event[0].replace("and", "--")
        split_statements = split_statements.replace("or", "--")
        split_statements = [x.strip() for x in split_statements.split("--")]
        
        times = []
        attribute_values = []
        for statement in split_statements:
            # Now, iterate through each statement and get its time and attribute value
            time, attr_value = self.get_attributes(statement, attribute)
            times.append(time)
            attribute_values.append(attr_value)
            
        time_attr_combos = self.gen_combinations(attribute_values, times)
        
        combination_statements = []
        # Now, we must iterate through every combination and create the equivalent statement
        for combo in time_attr_combos:
            completed_statement = \
                self.generate_statement_from_combo(combo, attribute, split_statements[0])
            combination_statements.append("("+completed_statement+")")
            
        combo_statement = ' or '.join(combination_statements)
        self.event = [combo_statement]

class Event:
    
    def __init__(self, event_str):
        self.event = [event_str]
    
class complexEvent:

    # Set up complex event
    def __init__(self):
#         self.states = states
#         self.transitions = transitions
#         self.ce_name = event_name
#         self.machine = None
          
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        event_call_text = text.split("=")[0]
        self.event_name = event_call_text
        
        
        self.watchboxes = {}
        self.config_watchboxes = {}
        self.client_info = {}
        self.executable_functions = []  # This is a list of [func_name, function]
        # This stores the previous state of each event initially
        self.previous_eval_result = {}  
        self.current_index = 0

        self.within_history = {}
        
        # Keep track of our json output
        self.result_output = {"incoming":[], "events":[]}
        self.result_dir = ""
        
        
    # Add watchboxes
    # def addWatchbox(self, **kwargs):
    def addWatchbox(self, event_content):
#         (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
#         event_call_text = text
        
#         # This is going to be the entire function call, so do some parsing
#         current_func_name = inspect.stack()[0][3]
#         # Filter out the function call
#         event_content = event_call_text.split(current_func_name)[1]
#         event_content = event_content[1:-1]
        
        # Now, initialize and remember this watchbox
        # This is done by executing the event as a string and saving it to our internal object list
        watchbox_name = event_content.split("=")[0].strip()
        watchbox_name_str = watchbox_name
        current_data = [watchbox_name]
        exec(event_content)
        exec("current_data.append(" + watchbox_name + ")")
        self.watchboxes.update({current_data[0]:current_data[1]})

        
    # Iterate through every watchbox name, and replace it with the object reference
    def replaceWatchboxCommands(self, event):
        
        # print("wbreplace")
        # print(event)
        new_commands = []

        # Iterate through every 'event'
        #  ev could be an event string, or it could be a list of event strings
        for ev in event:
            
            if type(ev) == str:
                ev = [ev]

            sub_new_commands = []

            # Itereate through sub ev
            for sub_ev in ev:
                new_command = sub_ev

                # print("---")
                # print(new_command)

                for wb in self.watchboxes.keys():
                    # If the watchbox name matches, replace the command with the string
                    if wb in new_command:
                        new_wb_object = "self.watchboxes[\""+wb+"\"]"
                        new_command = new_command.replace(wb, new_wb_object)
                
                # Append after making these changes
                sub_new_commands.append(new_command)
            
            new_commands.append(sub_new_commands)

        # print("\nnew command")
        # print(new_commands)
        
        return new_commands
    
    # Figure out how much 'history' each object needs to recall
    def updateMaxHistory(self, function_to_execute):

        print("here...")
        

        if type(function_to_execute) == list:
            function_to_execute = function_to_execute[0][0]
        
        
        # Get every watchbox involved, and their 'time' for looking back
        # Split by and/or
        watchbox_statements = function_to_execute.replace("and", "--")
        watchbox_statements = watchbox_statements.replace("or", "--")
        # Now split by the delimiter --
        split_statements = watchbox_statements.split("--")
        # Now iterate through each watchbox statement and figure out its history
        for x in split_statements:
            # Get the name of the watchbox
            name = x.split("[\"")[1].split("\"]")[0]
            history_len = int(x.split("at=")[1].split(",")[0])
            # Update the max history (checking of 'max' is done at the watchbox side)
            # self.watchboxes[name].updateMaxHistory(history_len)

    # For a sequence of split event names, make sure to actually connect the OR/AND events back together
    #  e.g. given ["OR(ev11a", " ev11b)"] turn it into ["OR(ev11a, ev11b)"]
    def connectOperatorEvents(self, split_sequence):
        buffer = []
        output = []
        start_buffer = False
        using_gen_permute = False
        skip_next_item = False
        # print(split_sequence)
        for x in split_sequence:
            
            if "(" in x:
                
                if "GEN_PERMUTE" in x:
                    using_gen_permute = True
                
                start_buffer = True
                x = x.split("(")[1]
                
                
                
            elif ")" in x:
                start_buffer = False
                x = x.split(")")[0]
                
                if using_gen_permute:
                    using_gen_permute = False
                    skip_next_item = True
                
            else:
                start_buffer = False
            

            if not skip_next_item:
                buffer.append(x)
            skip_next_item = False

            if not start_buffer:
                output.append(','.join(buffer))
                buffer = []

        return output
            

    
    
    # Add sequence of events

    # This initializes self.executable_functions to a particular format:
    #  [ 
    #    (   [ eventname, event_logic ] , event_operator ) ...
    #  ]
    # It also initializes self.previous_eval_result to a particular format:
    #  {
    #    eventname : [boolean state, number of times state became true]
    #  }
    def addEventSequence(self, events, no_enforce_sequence=False):
        
        # Also, get the event names
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        event_names = text.split("[")[1].split("]")[0].split(",")
        event_names = self.connectOperatorEvents(event_names)

        event_list = []
        # Do we enforce sequence constraints or no?
        self.no_enforce_sequence = no_enforce_sequence
        
        # Get each event
        for i,event in enumerate(events):

            # print("\n event name:")
            # print(event_names[i])
            
            # Check if this event is either a list or a string.  If it's a string, it is a single event, and otherwise as a list it is a multi-event involving AND/OR
            # Firstly, split and see if there is a comma here
            event_name_split = [x.strip() for x in event_names[i].split(",")]
            
            # If there are more than one events, then we actually have an operator involved.
            #  Otherwise, we just append the normal event and its name
            functions_to_execute = []
            for event_index in range(len(event_name_split)):
            # for event_index in range(len(event.event)):

                # print("\n split")
                # print(event_name_split)
                # print(event_name_split[event_index])

                # For every event, you must first replace it with our own local variables
                current_function = self.replaceWatchboxCommands([event.event[event_index]])
                current_name = event_name_split[event_index]
                event_list.append(current_name)
                self.within_history[current_name] = []
                functions_to_execute.append([current_name, current_function])
                # Parse the function, and determine how much 'history' each object needs to 
                #  remember, and update the max history for that watchbox
                # self.updateMaxHistory(current_function)
                
                # Initialize our evaluation results
                self.previous_eval_result[current_name] = [False, 0]
                
            # Now, check the operator (is it OR/AND?)
            operator = ""
            if len(functions_to_execute)>1:
                operator = event.event[-1]

            
            self.executable_functions.append((functions_to_execute, operator))
            
        # Initialize some results
        # eval_initial_results = []
        # for i in range(len(self.executable_functions)):
        #     current_functions = self.executable_functions[i][0]
        #     current_operator = self.executable_functions[i][1]
            
        #      # Now, evaluate each function
        #     current_eval_results = [False for x in current_functions]
        #     current_eval_functions = [x[0] for x in current_functions]

        #     eval_initial_results.append([current_eval_functions, current_eval_results, current_operator, False])
        # Initialize our visualization
        # self.visualize(eval_initial_results)
        return event_list
          
        

    # Add the compilation
    def compile(self):
        print("TBD")
          
#         # Basically, we construct our machine from the states and transitions
#         self.machine = HierarchicalMachine(model=self, states=self.states, \
#                 transitions=self.transitions, initial='asleep')

#         # We need to add a transition for going from 'asleep' to our first state.
        
    # Add update and evaluate
    def update(self, data):
        
        # Important note - sometimes, the update will involve results from two different timestamps
        # Like this: [{'camera_id': '2', 'results': [[[1], [True], 19974], [[0], [True], 20021]]
        #  We have to split it up appropriately.
        for result in data[0]["results"]:

            # Make sure we create a new result
            update_data = [{'camera_id': data[0]['camera_id'], 'results': [result]}]

            for x in self.watchboxes.keys():
                self.watchboxes[x].update(update_data)
                # print(x)
                # Now, print out every watchbox and its count
                # print([y.size for y in self.watchboxes[x].data])

    # This works by taking in previous class states and their history from the CE class
    def within(self, event_list, func_name, event_history, time_bound, time_index):
  
        # First, update our within history
        # event_history is a list of (event_index, time_index) representing when an event
        #  in event_list became true and at what time.
        # Evaluate every item in the ce list.  If true, update the ce history
        for ev_i, ce_item in enumerate(event_list):
            eval_result = eval(event_list[ev_i])
            print(event_list[ev_i])
            if eval_result:
                self.within_history[func_name].append((ev_i, float(time_index)))
        print("within history")
        print(self.within_history)

        # Then, look through our event_history to see if or when the last event became true
        all_events_occurred_on_time = [False for x in event_list[:-1]]
        event_index = len(event_list)-1
        last_event_time = -1
        # Look backwards in the event history
        print("looking at event_history")
        print(event_history)
        start_tracking = False
        for ev in event_history[::-1]:
            if ev[0] == len(event_list)-1: # THe last event in our within statement occurred
                start_tracking = True
                event_index -= 1
                last_event_time = ev[1]
                print("start tracking")
            else:
                if start_tracking: # If the last event occurred, 
                    print("current_ev_index:" + str(event_index))
                    print(ev)
                    if ev[0] == event_index: # If event index matches current index
                        if last_event_time - ev[1] < time_bound: #occurs within the time bound
                            all_events_occurred_on_time[event_index] = True
                            last_event_time = ev[1]
                            event_index -= 1
        print("Events occuring on time" + str(all_events_occurred_on_time))
        within_occurred = all(all_events_occurred_on_time)

        return within_occurred


    # Get the actual function we need to evaluate
    def get_func_to_evaluate(self, func_list, func_name, time_index):

        output = ""
        func_list = func_list[0] # Get rid of extra list
        if len(func_list) == 1:
            output = func_list[0]
        else:  # If we have multiple things to check, then it's likely we have an operator here.
            # Iterate through each function

            # If this is a within, then create the appropriate function name
            if func_list[-1][0] == "within":
                params = [str(func_list[0:-1]), "\'" + func_name + "\'"]
                params.extend(["self.within_history[\'"+ func_name +"\']", str(func_list[-1][2]), str(time_index)])

                output = "self.within(" + ','.join(params) + ")"

        return output


    def evaluate(self, time_index):
        
        change_of_state = False
        results = []
        
        eval_results = []
        old_results = []
        eval_indices_to_track = [self.current_index]

        # It is important to note how we are performing evaluation:
        #  Firstly, we obviously have to evaluate the 'current' state we are at in the FSM
        #  However, we must also evaluate certain previous states, as we wish to return an interval when they are true.
        #  So we evaluate everything up to the current state, and for previous states we evaluate them until they become false from true.
        for i in range(0, len(self.executable_functions)):
            
            # Otherwise, get the current function to execute
            # Remember, we have multiple functions to execute
            
            current_functions = self.executable_functions[i][0]
            current_operator = self.executable_functions[i][1]
            
            # Now, evaluate each function
            current_eval_results = [False for x in current_functions]
            current_eval_functions = [x[0] for x in current_functions]
            set_eval_result = False
            if i < self.current_index+1 or self.no_enforce_sequence:
                for f_i, function in enumerate(current_functions):

                    current_function_name = function[0]
                    current_function = function[1]

                    # # We avoid checking previous results if they already turned false from true
                    # if not self.previous_eval_result[current_function_name][0] and self.previous_eval_result[current_function_name][1] > 0:
                    #     continue

                    
                    current_function = self.get_func_to_evaluate(current_function, current_function_name, time_index)
                   
                    # If evaluation ends up with an error, spit out the error and return false
                    try:
                        eval_result = eval(current_function)
                        print("evaluating: " + current_function_name)
                        print("result: " + str(eval_result))
                    except Exception as e:
                        print(traceback.format_exc())
                        eval_result = False
                        print("Error in evaluation - returning false and continuing...")

                    current_eval_results[f_i] = eval_result
                    # current_eval_results.append(eval_result)
                    # current_eval_functions.append(current_function_name)

                    # Determine if a state has changed...
                    if self.previous_eval_result[current_function_name][0] != eval_result:
                        change_of_state = True
                        # results.append((current_function_name, eval_result))
                        self.previous_eval_result[current_function_name][0] = eval_result
                        self.previous_eval_result[current_function_name][1]+=1
                        old_results.append((current_function_name, eval_result))


                        # If we are not enforcing any particular order, then the current index is the index of this result
                        if self.no_enforce_sequence:
                            eval_indices_to_track.append(i)

                        
                    
                # Now, go back to the current function in the sequence (involving operators)
                #  Do the statements align with the operators?
                #  OR becomes 'any', and AND becomes 'all'

                if current_operator == "or":
                    set_eval_result = any(current_eval_results)
                elif current_operator == "and":
                    set_eval_result = all(current_eval_results)
                elif current_operator == "":  # Same as AND since we only have one result here.
                    set_eval_result = all(current_eval_results)
                
            eval_results.append([current_eval_functions, current_eval_results, current_operator, set_eval_result])
                
        # Get the current result
        current_result = []
        for eval_index in eval_indices_to_track:
            if eval_index < len(eval_results):
                current_result.append(eval_results[eval_index])
                    
                # If the most recent event has occurred, move up our window
                if self.current_index < len(eval_results) and eval_results[self.current_index][-1]:
                    self.current_index += 1
        
            
        #  Output of eval_results looks like this:
        # [
        #    [['ev11a', 'ev11a'], [True, True], 'or', True], 
        #    [['ev11b'], [False], '', False], 
        #    [['ev11c1'], [False], '', False], 
        #    [['ev11c2', 'ev11d'], [True, True], 'and', True]]
        # ]

        # Now, visualize our outputs
        # self.visualize(eval_results)
            
#         # If the current index reaches the max, then the final event has occurred
#         if self.current_index >= len(self.executable_functions):
#             results.append((self.event_name, True))
            
#         return change_of_state, results

        return current_result, change_of_state, old_results
        
        
          
          
    # Add the visualization
    def visualize(self, eval_results):
        
        print(eval_results)

        # m = Model()
        # # without further arguments pygraphviz will be used
        # machine = HierarchicalGraphMachine(model=m, states=self.states, \
        #         transitions=self.transitions, initial='asleep')
        # # draw the whole graph ...
        # m.get_graph().draw('my_state_diagram.png', prog='dot')
        
        # First, we have to create an empty image where height is proportionate to 
        #  the number of states
        num_blocks = len(eval_results)
        blank_image = np.zeros((num_blocks*100,500,3), np.uint8)
        blank_image.fill(255) # or img[:] = 255
        
        # Initialize a set of boxes 
        rpos = [[0,0], [500,100]]
        gtext_pos = [250, 20]
        for general_event in eval_results:
            
            # General event name and result
            general_event_name = general_event[2]
            general_event_result = general_event[3]
            # Sub events and their results
            sub_events = general_event[0]
            sub_event_results = general_event[1]
            
            cv2.rectangle(blank_image, rpos[0], rpos[1], color=(0,255,0), thickness=3)
            rpos[0][1] += 100
            rpos[1][1] += 100
            
            cv2.putText(blank_image, general_event_name, gtext_pos, cv2.FONT_HERSHEY_SIMPLEX,\
                        1, (255, 0, 0), 2, cv2.LINE_AA)
            gtext_pos[1] += 100
            
        
            # Now, draw the inner rectangles
            divided_amount = rpos[1][0] // len(sub_events)
            sub_rect_position = [[0,rpos[0][1]+20],[divided_amount,rpos[1][1]]]
            for sub_event_i in range(len(sub_events)):
                
                sub_event_name = sub_events[sub_event_i]
                sub_event_result = sub_event_results[sub_event_i]
                # Place boxes and text
                cv2.rectangle(blank_image, sub_rect_position[0]\
                                , sub_rect_position[1], color=(255,255,0), thickness=2)
                sub_text_pos = sub_rect_position[0]
                sub_text_pos[0] -= 20
                sub_text_pos[0] += divided_amount//2
                cv2.putText(blank_image, general_event_name, sub_text_pos, cv2.FONT_HERSHEY_SIMPLEX,\
                        1, (255, 0, 0), 2, cv2.LINE_AA)
                
                # Update new positions
                sub_rect_position[0][0] += divided_amount
                sub_rect_position[1][0] += divided_amount
                
        
        cv2.imshow("frame", blank_image)


# def ce_and(event_list, time_constraints):

#     (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
#     ce_name = text[:text.find('=')].strip()

#     # Name the current state
#     #  Either "AND" is satisfied or it is not.

#     sub_states = []
#     sub_transitions = []
#     for event in event_list:
#         sub_states.extend(event.states)
#         sub_transitions.extend(event.transitions)

#     states = [
#         {'name': ce_name + '-true-', 'children':sub_states},
#         {'name': ce_name + '-false-', 'children':sub_states},
#     ]
#     transitions = [
#         {"trigger": ce_name + '-and',
#             "source": ce_name + '-false',
#             "dest": ce_name + '-true' },
#         {"trigger": ce_name + '-notand',
#             "source": ce_name + '-true',
#             "dest": ce_name + '-false' }
#     ]
#     transitions.extend(sub_transitions)


#     return complexEvent(states, transitions, ce_name)


# def ce_until(event_list, time_constraints):


#     return

# def ce_follows(event_list):

#     event_name = get_param_name(event_list)

#     return



# The first thing we need to do is to be able to construct a tree of 
#  the functions and their order of execution.  This means constructing
#   a list of states (ALL STATES.)