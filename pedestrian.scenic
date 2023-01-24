#################################
# MAP AND MODEL                 #
#################################

from carla_code.draw_reference import read_data
import pdb
import math
import random


import sys
try:
	sys.path.append('CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg')
	sys.path.append('CARLA_0.9.10/PythonAPI/carla/')
	import carla as _carla
except ImportError as e:
	raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

from carla_code.object_dummy import Object_Dummy
import scenic.simulators.carla.utils.utils as utils
import scenic.simulators.carla.blueprints as blueprints

param map = localPath('/home/brianw/Documents/PTZCameraRecognition/CARLA_0.9.10/CarlaUE4/Content/Carla/Maps/OpenDrive/Town10HD.xodr')
param carla_map = 'Town10HD'
model scenic.simulators.carla.model



fps = 10


global_state = 0
obj_dumm = None

#################################
# AGENT BEHAVIORS               #
#################################

behavior WalkAround(destination):
    global fps
    
    if self.carlaController:
        state = 0
        past_time = 0
        #destination = Uniform(10, -10)*destination
        take SetWalkAction(True)
        take SetWalkDestination(destination)
        threshold = Uniform(1,120)*fps
        while True:
            if state == 0:
                d = distance from self to destination
                if d < 2:
                    take SetWalkingSpeedAction(0)
                    state += 1
                    past_time = simulation().currentTime
                else:
                    wait
            elif state == 1 and simulation().currentTime - past_time > threshold:
                destination = Range(-94.278351,-60.667870) @ Range(29.682917,51.475655) #Uniform(10, -10)*destination
                take SetWalkingSpeedAction(1.4)
                take SetWalkDestination(destination)
                state -= 1
            else:

                wait
            

behavior StealingPackagesBehaviorSingle():
    global fps
    state = 0
    past_time = 0
    take SetWalkAction(True)
    take SetWalkDestination(box)
    while True:
    
        if state == 0:
            d = distance from self to box
            if d < 2:
                take SetWalkingSpeedAction(0)
                state += 1
                past_time = simulation().currentTime
            else:
                take SetWalkingSpeedAction(1.4)
                #take SetWalkDestination(box)
        elif state == 1 and simulation().currentTime - past_time > Uniform(10,20)*fps:
            idx = actors_bb.index(box)
            del actors_bb[idx]
            box.destroy(simulation())
            

            state += 1
            past_time = simulation().currentTime
        elif state == 2 and simulation().currentTime - past_time > Uniform(1,5)*fps:
            take SetWalkingSpeedAction(1.4)
            state += 1
            past_time = simulation().currentTime
        elif state == 3 and simulation().currentTime - past_time > Uniform(10,60)*fps:
            terminate
            
        else:
            wait
   



behavior WaitAndGo(destination):
    global global_state
    
    local_state = 0
    while True:
        if global_state == 1 and local_state == 0:
            take SetWalkAction(True)
            take SetWalkDestination(destination)
            local_state += 1
            
        wait

behavior LeavingPackagesBehaviorSingle(box_destination):
    global fps, global_state
    state = 0
    past_time = 0
    take SetWalkAction(True)
    take SetWalkDestination(box_destination)
    while True:
    
        if state == 0:
            d = distance from self to box_destination
            if d < 2:
                take SetWalkingSpeedAction(0)
                state += 1
                past_time = simulation().currentTime
            else:
                take SetWalkingSpeedAction(1.4)
                #take SetWalkDestination(box)
        elif state == 1 and simulation().currentTime - past_time > Uniform(10,20)*fps:
            obj_dumm = Object_Dummy(Uniform(*blueprints.boxModels),box_destination,0,0)
            simulation().createObjectInSimulator(obj_dumm)
            #simulation().objects.append(obj_dumm)
            actors_bb.append(obj_dumm)
            global_state += 1

            state += 1
            past_time = simulation().currentTime
        elif state == 2 and simulation().currentTime - past_time > 60*fps:
            state += 1
            past_time = simulation().currentTime
        elif state == 3 and simulation().currentTime - past_time > 60*fps:
            #take SetWalkingSpeedAction(1.4)
            take SetWalkDestination(67.793434 @ -3.075324)
            take SetWalkingSpeedAction(1.4)
            state += 1
            
            
        else:
            wait   
            
behavior StealingPackagesBehaviorSupplement(destination):
    global fps
    state = 0
    past_time = 0
    take SetWalkAction(True)
    take SetWalkDestination(destination)
    while True:
    
        if state == 0:
            d = distance from self to destination
            if d < 2:
                take SetWalkingSpeedAction(0)
                state += 1
                past_time = simulation().currentTime
                take SetWalkAction(False)
            else:
                take SetWalkingSpeedAction(1.4)

        elif state == 1 and simulation().currentTime - past_time > Uniform(10,20)*fps:

            
            d = distance from self to destination
            if d < 1:
                take SetWalkingSpeedAction2(0)
                idx = actors_bb.index(box2)
                del actors_bb[idx]
                box2.destroy(simulation())
      

                state += 1
                past_time = simulation().currentTime
                
            else:
                take SetWalkingSpeedAction2(1.4)
            

   
        elif state == 2 and simulation().currentTime - past_time > Uniform(1,5)*fps:
            take SetWalkAction(True)
            take SetWalkingSpeedAction(1.4)
            state += 1
            past_time = simulation().currentTime

            
        else:
            wait
            
behavior RandomActionPackage(destination):
    decision = Uniform(0,1)
    print(decision)
    #decision = 1
    if not decision:
        take SetWalkAction(True)
        while True:
            wait
    else:
        do StealingPackagesBehaviorSupplement(destination)
        
            
behavior StealingPackagesBehaviorSingleCar(destination,car):
    global fps,global_state
    state = 0

    past_time = 0
    take SetWalkAction(True)
    take SetWalkDestination(destination)
    while True:

        if state == 0:
            d = distance from self to destination
            if d < 2:
                take SetWalkingSpeedAction(0)
                take SetWalkAction(False)
                state += 1
                past_time = simulation().currentTime
            else:
                take SetWalkingSpeedAction(1.4)

                #take SetWalkDestination(box)
        elif state == 1 and simulation().currentTime - past_time > Uniform(10,20)*fps:
            d = distance from self to destination
            if d < 1:
                take SetWalkingSpeedAction2(0)
                idx = actors_bb.index(box)
                del actors_bb[idx]
                box.destroy(simulation())
                global_state += 0.5

                state += 1
                past_time = simulation().currentTime
                
            else:
                take SetWalkingSpeedAction2(1.4)
        elif global_state == 1 and state == 2 and simulation().currentTime - past_time > Uniform(1,5)*fps:
            #take SetWalkingSpeedAction(1.4)
            #take SetWalkDestination(car)
            #take SetWalkAction(False)
            take SetWalkingDirectionAction(angle from self to car)
            state += 1
        
        elif state == 3:
            #print(angle from self to car)
            
            take SetWalkingSpeedAction2(1.4)
            d = distance from self to car
            #print(d)
            if d < 2:
                global_state += 1
                state += 1
                idx = actors_bb.index(self)
                del actors_bb[idx]
                self.destroy(simulation())

                
            else:
                wait
 
        
        else:
            wait

behavior StealingPackagesBehaviorDoubleA(destination):
    
    global obj_dumm,global_state
    take SetWalkAction(True)
    take SetWalkDestination(destination)
    state = 0
    past_time = 0
    while True:
        if state == 0:
            d = distance from self to destination
            if d < 3:
                take SetWalkingSpeedAction(0)
                global_state += 0.5
                state += 1
                past_time = simulation().currentTime
                obj_dumm = Object_Dummy(Uniform(*blueprints.boxModels),destination,0,0)
                simulation().createObjectInSimulator(obj_dumm)
                #simulation().objects.append(obj_dumm)
                actors_bb.append(obj_dumm)
                #print(global_state)
            else:
                take SetWalkingSpeedAction(1.4)
        elif state == 1 and global_state == 1:
            state += 1
            past_time = simulation().currentTime
        elif state == 2 and simulation().currentTime - past_time > Uniform(10,20)*fps:
            take SetWalkingSpeedAction(1.4)
            state += 1
        else:
            wait
    
behavior StealingPackagesBehaviorDoubleB(destination):
    
    global obj_dumm,global_state
    take SetWalkAction(True)
    take SetWalkDestination(destination)
    state = 0
    past_time = 0
    while True:
        if state == 0:
            d = distance from self to destination

            if d < 2:
                take SetWalkingSpeedAction(0)
                global_state += 0.5
                state += 1

            else:
                take SetWalkingSpeedAction(1.4)
        elif state == 1 and global_state == 1:
            state += 1
            past_time = simulation().currentTime
        elif state == 2 and simulation().currentTime - past_time > Uniform(10,20)*fps:
            idx = actors_bb.index(obj_dumm)
            del actors_bb[idx]
            obj_dumm.carlaActor.destroy()
            take SetWalkingSpeedAction(1.4)
            past_time = simulation().currentTime
            state += 1
        elif state == 3 and simulation().currentTime - past_time > Uniform(30,60)*fps:
            terminate
        else:
            wait
        

behavior CameraBehavior(path):
    while True:
        take GetBoundingBox(actors_bb,path)
		
behavior CarFollowingBehavior():

    try:
        do FollowLaneBehavior()
    interrupt when self.distanceToClosest(Pedestrian) < 20:
        take SetBrakeAction(1)
        
behavior CarFoBehavior(destination):
    
    global fps, global_state
    
    #position = box.carlaActor.get_transform().location
    position = utils.scenicToCarlaLocation(destination,world=simulation().world)
    agent = BasicAgent(self.carlaActor)
    map = simulation().map

    chosen_waypoint = map.get_waypoint(position,project_to_road=True, lane_type=_carla.LaneType.Driving)
    current_waypoint = map.get_waypoint(self.carlaActor.get_transform().location,project_to_road=True, lane_type=_carla.LaneType.Driving)
    new_route_trace = agent._trace_route(current_waypoint, chosen_waypoint)
    #pdb.set_trace()
    #print(new_route_trace)
    agent._local_planner.set_global_plan(new_route_trace)
    self.carlaActor.apply_control(agent.run_step())
    state = 0
    past_time = 0
    while True:            
        if state == 0:	       
            state += 1
        elif state == 1:
            control = agent._local_planner.run_step()
            self.carlaActor.apply_control(control)
            
            if agent.done():
                state += 1
                past_time = simulation().currentTime
                agent._target_speed = 0
                self.carlaActor.apply_control(agent.run_step())
                print("done")
                global_state += 0.5
        elif global_state == 2 and state == 2 and simulation().currentTime - past_time > Uniform(10,20)*fps:
            past_time = simulation().currentTime
            print("followLane")
            try:
                do FollowLaneBehavior()
            interrupt when simulation().currentTime - past_time > Uniform(10,20)*fps:
                terminate
            
        
        wait
    
behavior CarFollowBehavior():
    #take GetPathVehicle(self,box)
    
    map = simulation().map
    agent = BasicAgent(self.carlaActor)
    position = box.carlaActor.get_transform().location
    chosen_waypoint = map.get_waypoint(position,project_to_road=True, lane_type=_carla.LaneType.Driving)
    current_waypoint = map.get_waypoint(self.carlaActor.get_transform().location,project_to_road=True, lane_type=_carla.LaneType.Driving)
    new_route_trace = agent._trace_route(current_waypoint, chosen_waypoint)

    route = []
    network_lanes = [n.id for n in network.lanes]
    road_lanes = {}
    for n in network.lanes:
        r_id = n.road.id
        if r_id in road_lanes:
            road_lanes[r_id] += 1
        else:
            road_lanes[r_id] = 1
 
    
    waypoints = simulation().map.generate_waypoints(distance=2.0)
    # Now we take our waypoints generated from the map, and get the corresponding roads.
    # road_dict is {road_id: [waypoint, waypoint, etc.]}
    road_dict = {}
    for wp in waypoints:
        if wp.road_id not in road_dict:
            road_dict[wp.road_id] = []
        if wp.lane_id not in road_dict[wp.road_id]:
            road_dict[wp.road_id].append(wp.lane_id)
        
    #for i in road_dict.keys():
    #    print(i,road_dict[i])

    idx = 0
    last_idx = -1
    #print(road_lanes)
    count = 0
    for i in new_route_trace:
        print(i[0].road_id,i[0].section_id,i[0].lane_id,i[0].s)
        
        lane_id = i[0].lane_id+math.floor(road_lanes[i[0].road_id]/2.0)
        
        if road_lanes[i[0].road_id] == 1:
            lane_id = 0
        elif i[0].lane_id > 0:
            lane_id -= 1
        

        lane_id = road_dict[i[0].road_id].index(i[0].lane_id)
       
        road_name = 'road'+str(i[0].road_id)+'_'+'lane'+str(lane_id)
        
        if road_name in network_lanes:
            idx = network_lanes.index(road_name)

            if idx != last_idx:
                if count == 1:
                    del route[-1]
                route.append(network.lanes[idx])
                count = 0
            count += 1
            last_idx = idx
        else:
            print("bad", road_name, i[0].lane_id, i[0].road_id, road_lanes[i[0].road_id], math.floor(road_lanes[i[0].road_id]/2.0))
        
    #take TrackWaypointsAction(new_route_trace)
    print(route)
    try:
        do FollowTrajectoryBehavior(trajectory=route,target_speed=5)
    interrupt when self.distanceToClosest(Box) < 20:
        take SetBrakeAction(1)
        print("yaa")

    #while True:
    #    wait
    
    
behavior BombScare(destination,box_destination):
    global global_state
    
    try:
        do WalkAround(destination)
    interrupt when global_state == 1:
        take SetWalkAction(False)
        #print(angle from self to box_destination)
        take SetWalkingDirectionAction(math.pi/2 - (angle from self to box_destination))
        take SetWalkingSpeedAction2(3)
        while True:
            wait
        
behavior Terrorist(destination):
    
    global obj_dumm,global_state
    take SetWalkAction(True)
    take SetWalkDestination(destination)
    state = 0
    past_time = 0
    while True:
        if state == 0:
            d = distance from self to destination
            if d < 2:
                take SetWalkingSpeedAction(0)
                
                state += 1
                past_time = simulation().currentTime
                obj_dumm = Object_Dummy(Uniform(*blueprints.boxModels),destination,0,0)
                simulation().createObjectInSimulator(obj_dumm)
                #simulation().objects.append(obj_dumm)
                actors_bb.append(obj_dumm)
                #print(global_state)
            else:
                take SetWalkingSpeedAction(1.4)
        elif state == 1:
            state += 1
            past_time = simulation().currentTime
        elif state == 2 and simulation().currentTime - past_time > Uniform(10,20)*fps:
            take SetWalkingSpeedAction(1.4)
            global_state += 1
            state += 1
        else:
            wait    
#################################
# SCENARIO SPECIFICATION        #
#################################

#for i in network.lanes:
#    print(i.id)



    

"""
select_lane = ""
for lane in network.lanes:
    if lane.id == "road5_lane1":
        select_lane = lane
"""

destination_locations = [Range(-94.278351,-60.667870) @ Range(29.682917,51.475655),Range(66.135605,93.087204) @ Range(-5.828710,14.528165),Range(-11.539524,20.287785) @ Range(74.251755,108.597641),Range(-138.710266,-122.311989) @ Range(60.340248,85.676659)]
box_destination = Uniform(*destination_locations)

num_pedestrians = 100

walkerModels = blueprints.walkerModels



#Second scenario
#car = Car with behavior CarFoBehavior(box_destination)
    #with behavior FollowTrajectoryBehavior(trajectory=[network.lanes[0],network.lanes[38]])


box = Box at box_destination #Range(-10,1) @ Range(-5,-5.5) #19.195606 @ -2.534948 #39.821236 @ 11.909901 #box_destination

box_destination2 = Uniform(*destination_locations)

box2 = Box at box_destination2

#Box at Uniform(*destination_locations)

bluep = walkerModels[0] #random.choice(walkerModels)

"""
ego = Pedestrian in sidewalk, #at -94.900848 @ 24.582340,
    with behavior StealingPackagesBehaviorSingleCar(box_destination,car),
    with blueprint bluep
"""
    
walkerModels.remove(bluep)

ego = Pedestrian at 67.793434 @ -3.075324,
    with behavior LeavingPackagesBehaviorSingle(Range(-10,1) @ Range(-5,-5.5)), #RandomActionPackage(box_destination2),
    with blueprint random.choice(walkerModels)
    
ped2 = Pedestrian at 70.793434 @ -3.075324,
    with behavior WaitAndGo(-31.996809 @ -3.644337),
    with blueprint random.choice(walkerModels)

#actors_bb = [ego,box,car,ped2,box2]
actors_bb = [ego, ped2]


divisor = num_pedestrians / len(destination_locations)
destination_idx = 0
divisor_mult = divisor

"""
for i in range(num_pedestrians):
    if i >= divisor_mult:
        destination_idx +=1
        divisor_mult = divisor*(destination_idx+1)
    
    ped = Pedestrian in sidewalk,
        with behavior WalkAround(destination_locations[destination_idx]),
        with blueprint random.choice(walkerModels)

        
    actors_bb.append(ped)

"""
"""
##First scenario 

    
bluep = random.choice(walkerModels)
two_ped = Pedestrian  in sidewalk, #at -94.900848 @ 24.582340,
    with behavior StealingPackagesBehaviorDoubleA(box_destination),
    with blueprint bluep
    
walkerModels.remove(bluep)
bluep = random.choice(walkerModels)
ego = Pedestrian in sidewalk, #at -59.610615 @ 23.937918,
    with behavior StealingPackagesBehaviorDoubleB(box_destination),
    with blueprint bluep


walkerModels.remove(bluep)
#box = Box at Range(-94.278351,-60.667870) @ Range(29.682917,51.475655)

destination = -75.037537 @ 49.348728


actors_bb = [ego,two_ped]
"""

"""
for i in range(100):
    ped = Pedestrian in sidewalk,
    with behavior WalkAround(Range(-94.278351,-60.667870) @ Range(29.682917,51.475655)),
    with blueprint random.choice(walkerModels)

    #if ped.carlaActor is not None:
    actors_bb.append(ped)
"""        
	       




"""
for i in range(100):
    ped = Pedestrian in sidewalk,
        with behavior WalkAround(Range(-94.278351,-60.667870) @ Range(29.682917,51.475655)),
        with blueprint random.choice(walkerModels)

    #if ped.carlaActor is not None:
    actors_bb.append(ped)
"""

"""
##Fourth scenario





for i in range(100):
    if i < 50:
        ped = Pedestrian in sidewalk,
            with behavior WalkAround(Range(-94.278351,-60.667870) @ Range(29.682917,51.475655)),
            with blueprint random.choice(walkerModels)
    else:
        ped = Pedestrian in sidewalk,
            with behavior WalkAround(Range(66.135605,93.087204) @ Range(-5.828710,14.528165)),
            with blueprint random.choice(walkerModels)

    #if ped.carlaActor is not None:
    actors_bb.append(ped)
"""

"""
##Third scenario

bluep = random.choice(walkerModels)
ego = Pedestrian in sidewalk,
    with behavior Terrorist(box_destination),
    with blueprint bluep

actors_bb = [ego]
walkerModels.remove(bluep)
for i in range(100):
    ped = Pedestrian in sidewalk,
        with behavior BombScare(Range(-94.278351,-60.667870) @ Range(29.682917,51.475655),box_destination),
        with blueprint random.choice(walkerModels)

    #if ped.carlaActor is not None:
    actors_bb.append(ped)

"""

"""
box = Box at -9.937593 @ -79.820206,
    with elevation 1
"""

"""
points_data = read_data("carla_code/points.txt")
camera_descriptions = [x for x in points_data if "tc" in x[-1]]

for c in camera_descriptions:

    depth_camera = depthCamera at c[0] @ -c[1],
        with elevation c[2],
        with pitch c[3],
        with yaw c[4],
        with roll 0

    rgbCamera at depth_camera,
        with elevation c[2],
        with pitch c[3],
        with yaw c[4],
        with roll 0,
        with depth depth_camera, 
        with camera_id int(c[-1][-1]),
        with behavior CameraBehavior("camera_img/")


"""		

