'''
Circle Example
==============

This example exercises circle (ellipse) drawing. You should see sliders at the
top of the screen with the Kivy logo below it. The sliders control the
angle start and stop and the height and width scales. There is a button
to reset the sliders. The logo used for the circle's background image is
from the kivy/data directory. The entire example is coded in the
kv language description.
'''

from kivy.app import App
from kivy.graphics import *
from kivy.graphics import RoundedRectangle
from kivy.uix.widget import Widget
from kivy.clock import Clock

from kivy.core.window import Window
from kivy.uix.label import Label

# Add arrows
from kivyarrow.arrow import *

from run_experiment import execute_main_experiment

import time
import threading
from random import random
from queue import Queue
import socket


class CanvasWidget(Widget):

    def __init__(self, structure):
        super(CanvasWidget, self).__init__()
        self.structure = structure


        # Get the total length of all characters
        self.charsum = sum([len(x) for x in self.structure])





    # Now, build our FSM based on the structure
    # def build(self, event_states):
    #     print(self.structure)

    #     with self.canvas:

    #         # Next, loop through every item in our structure
    #         box_position = (10,10)
    #         box_width = 100
    #         margin = 20
    #         for ev_name in self.structure:

    #             # Add a red color
    #             x = Color(1.,0, 0)

    #             # Add a rectangle
    #             Rectangle(pos=box_position, size=(box_width, box_width))

    #             # For some odd reason, a label is always missed the first time
    #             #  So it has to get added twice???
    #             new_label = Label()
    #             new_label.pos = box_position
    #             new_label.text = ev_name
    #             new_label = Label()
    #             new_label.text = ev_name
    #             new_label.pos = box_position
                
    #             # Update the positions
    #             box_position = (box_position[0]+box_width + margin, box_position[1])

    #     return self
    
    #  event_occurrence_list is actually a list of the following:
    #  [(event_name, truth_value)]
    def update(self, event_occurrence_list):

        with self.canvas:

            # Next, loop through every item in our structure
            box_position = (10,10)
            box_padding = 20
            box_height = 100
            margin = 20
            arrows_to_add = []
            labels_to_add = []
            for i,event in enumerate(event_occurrence_list):

                text_width = len(event[0])*10

                # Add a red color if false, green if true
                if event[1]:
                    x = Color(0, 0.5, 0)      # Box should be green ish
                    text_color = (1,1,1,1)  # Text should be white is
                else:
                    x = Color(0.6, 0,0)      # Box should be red ish
                    text_color = (1,1,1,1)  # Text should be white ish

                # Add a rectangle
                RoundedRectangle(pos=box_position, size=(text_width+box_padding, box_height))

                # For some odd reason, a label is always missed the first time
                #  So it has to get added twice???
                new_label = Label(font_size='15sp', markup=True)
                new_label.pos = (box_position[0]+(text_width//4), box_position[1])
                new_label.text = event[0]
                new_label.color = text_color
                labels_to_add.append(new_label)
                
                # Update the positions
                box_position = (box_position[0]+ text_width + box_padding + margin, box_position[1])

                # Only add an arrow if this is not the last event
                if i < len(event_occurrence_list)-1:
                    newarrow = Arrow(
                        main_color=[0.05, 0.33, 0.87,0.8],
                        outline_color=[0,0,1,0.8],
                        o_x= box_position[0]-30, 
                        o_y= 60,
                        #to_x=random()*self.layout.width,
                        #to_y=random()*self.layout.height,
                        # angle=random()*30,
                        distance=40,#+(random()*100),
                        fletching_radius=cm(0.2),
                        # distortions=[random() * -0.2, random() *0.3] if random()>0.5 else [],
                        # head_angle=40+(random()*60)
                    )
                    arrows_to_add.append(newarrow)

            # For every label, add it.
            for lbl in labels_to_add:
                self.add_widget(lbl)
            # For every arrow, add it.  This brings all arrows to the 'front'
            for arrow in arrows_to_add:
                self.add_widget(arrow)
            

        return self
    

# Create the application
class CE_display(App):

    Window.clearcolor = (1, 1, 1, 1)

    def __init__(self, queue, ce_structure):
        super(CE_display, self).__init__()
        self.event_queue = queue
        self.ce_structure = ce_structure
        self.event_states = [[x, False] for x in self.ce_structure]
        self.event_states.append(["Event Occurred", False])

        server_listen_thread = threading.Thread(target=self.listen_for_updates)
        server_listen_thread.start()


    # Listen to a server thread
    def listen_for_updates(self):

        server_addr = ("127.0.0.1", 8001)
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        serverSocket.bind(server_addr)
    
        print("Listening on " + str(server_addr))

        # Now for the recv logic
        recv_config = False
        while True:
            try:
                data, addr = serverSocket.recvfrom(512)
                data = data.decode()
                print(data)
                data = eval(data)
                # First, eval the data, as it will provide us with something
                if not recv_config:
                    print("Received config.")
                    self.event_states = [[x, False] for x in data]
                    self.event_states.append(["Event Occurred", False])
                    recv_config = True
                else:  # We already recieved the config, so now just get events
                    print("RECEIVING DATA")
                    print(data)
                    self.event_queue.append(data)

            except Exception as e:
                print(e)
                input()

    def build(self):
        self.widget = CanvasWidget(self.event_states)

        # Based on the size, determine how big the window should be
        Window.size = (self.widget.charsum*160, 120)


        Clock.schedule_interval(self.update, 0.25)
        return self.widget.update(self.event_states)

    def update(self, *args):
        
        # We need to parse the event queue
        # print(self.event_queue)
        # print(self.event_states)
        # self.widget.update(self.event_states)
        
        # Get the event that occurred
        # If the event queue actually has something:
        if self.event_queue:
            print(self.event_queue)
            event = self.event_queue[0][0][-1]  # Something like ('ev11a', True)
            returned_ev_name = event[0]
            ev_updated_state = event[1]
            # Get the index of this event in our structure, and alter the event state
            for i,ev_state in enumerate(self.event_states):

                if returned_ev_name == ev_state[0]:
                    self.event_states[i][1] = ev_updated_state

                    # Note - if the final event occurs, then we need to 
                    #  - clear previous events, and declare the final event occuring
                    if i == len(self.event_states)-2:
                        self.event_states[-1][1] = True
                        # for j in range(0,len(self.event_states)-2):
                        #     self.event_states[j][1] = False
             
            
            self.event_queue.pop()
            # self.event_queue = [] # Empty the list
        self.widget.update(self.event_states)


if __name__ == '__main__':


    event_list = []
    ce_structure = []  # This is a list of strings for each CE

    # Run teh other thread
    t = threading.Thread(target=execute_main_experiment)
    t.start()

    # Wait a bit for the structure to get set up
    # while not ce_structure:
    #     pass

    # Run the application
    fsm_visualapp = CE_display(event_list, ce_structure)
    fsm_visualapp.run()

# Firstly, run the 
    



