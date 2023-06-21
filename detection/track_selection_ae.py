import os
import cv2
import pybboxes as pbx
import pdb
import numpy as np
from collections import defaultdict
from itertools import combinations, permutations
import argparse



#Select tracks based on context

def sort_file(e):
	return int(e[:-4])


"""
Parse file with vicinal events annotations
aes format is the following {ground_truth track id:[[camera_id,watchbox,enter_frame,exit_frame], ...], ...}
If there is no exit frame registered, a -1 is put in the exit_frame
"""
def process_ae_file(ae_file):
	ae_f = open(ae_file)
	ae_lines = ae_f.readlines()
	
	ae_array = {}
	prev_camera = {}
	for ae_line in ae_lines:
		ae_temp = eval(ae_line.strip().split(":::")[1])
		
		for ae in ae_temp:
			for res in ae['results']:			
				if res[2] not in ae_array:
					ae_array[res[2]] = []
				if res[2] not in prev_camera:
					prev_camera[res[2]] = -1
					
					
				if prev_camera[res[2]] != ae['camera_id']:
					for ae_times_idx, ae_times in enumerate(ae_array[res[2]]):
						if ae_times[0] == prev_camera[res[2]] and ae_times[3] < 0:
							ae_array[res[2]][ae_times_idx][3] = ae['time']-1
							

						
				for w_idx,watchbox in enumerate(res[0]):
				
					if ae['camera_id'] == '2' and watchbox == 0: #Ignore big watchbox that overlaps with all the other ones
						continue
				
					ae_idx = -1
					for ae_times_idx, ae_times in enumerate(ae_array[res[2]]):
						if ae_times[0] == ae['camera_id'] and ae_times[1] == watchbox:
							ae_idx = ae_times_idx
							
					if ae_idx > -1:
						if res[1][w_idx]:
							ae_array[res[2]][ae_idx][2] = ae['time']
						else:
							ae_array[res[2]][ae_idx][3] = ae['time']
					else:
						if res[1][w_idx]:
							ae_array[res[2]].append([ae['camera_id'], watchbox, ae['time'], -1])
						else:
							ae_array[res[2]].append([ae['camera_id'], watchbox, 0, ae['time']])
							
				prev_camera[res[2]] = ae['camera_id']
	return ae_array
	
	
def FindPoints(watchbox, point): #Check if points are inside a watchbox

	conditions = (point[:,0] > watchbox[0]) & (point[:,0] < watchbox[2]) & (point[:,1] > watchbox[1]) & (point[:,1] < watchbox[3])
	return conditions


def FindPoint(watchbox, point): #Check if a single point is inside a watchbox

	conditions = (point[0] > watchbox[0]) and (point[0] < watchbox[2]) and (point[1] > watchbox[1]) and (point[1] < watchbox[3])
	return conditions

#Not used
def recursive_search(graph_links, edge): 
			
	scores = []
	for e in graph_links["edge"][edge]:
		if graph_links["group_score"][e] >= 0:
			scores.append(graph_links["group_score"][e])
		else:
			scores.append(recursive_search(graph_links,e))
		
	if scores:
		result = graph_links["self_score"][edge] + max(scores)
	else:
		result = graph_links["self_score"][edge]
	return result
	
"""
Backwards pass. The way it works is that it will traverse the graph starting with the watchbox tracks associated with watchbox evt, for each node. It will first go to the nodes that have no more edges associated, get the matching score, and then go back the previous nodes and assign their score based on the maximum score one could obtain by following that path. A penalty is given to all nodes representing tracks that skip through watchboxes they should be present in according to the ground truth track.
"""
def recursive_search2(graph_links, evt, edge, penalty):
			
	scores = []
	for e_idx,e in enumerate(graph_links[evt]["edge"][edge]):
		cevt = graph_links[evt]["ae"][edge][e_idx]
		if graph_links[cevt]["group_score"][e] >= 0:
			scores.append(graph_links[cevt]["group_score"][e])
		else:
			scores.append(recursive_search2(graph_links,cevt,e,penalty))
		
	if scores:
		result = graph_links[evt]["self_score"][edge] + max(scores)
	else:
		result = graph_links[evt]["self_score"][edge] - penalty
	return result


"""
Function that prepares to get the shortest path, our approach is based on the topological sort algorithm
"""
def get_shortest_path(graph_links, number_vertices, removed_nodes=[]):


	number_to_key = {"0_-1":0}
	num_idx = 1
	


	terminal_nodes = []
	g = Graph(number_vertices-len(removed_nodes))



	for evt in graph_links.keys(): #First we build the graph using the addEdge function, as well as a number_to_key dictionary to allow us to associated a number to each subtrack as that is how the shortest path alogrithm expects nodes to be represented. If there are any nodes to be removed, this is were they are skipped. Also we build a list of terminal nodes, those without edges
		for track in graph_links[evt]["final"].keys():
		
				
			key_track = str(track)+"_"+str(evt)
			
			
			
			if key_track in removed_nodes:
				continue
			

				
			
				
			
			if key_track not in number_to_key:
				number_to_key[key_track] = num_idx
				num_idx += 1
			
			
			real_nodes = []
			
			for nt_idx,next_track in enumerate(graph_links[evt]["edge"][track]):

				key_track2 = str(next_track)+"_"+str(graph_links[evt]["ae"][track][nt_idx])
				
				if key_track2 not in removed_nodes:
				
					real_nodes.append(key_track2)
					
					if key_track2 not in number_to_key:
						number_to_key[key_track2] = num_idx
						num_idx += 1
						
					g.addEdge(number_to_key[key_track], number_to_key[key_track2], graph_links[evt]["final"][track][nt_idx])

			if not real_nodes:
			
				terminal_nodes.append(number_to_key[key_track])



	#print("Terminal nodes",terminal_nodes)



	paths,dist = g.shortestPath(0) #The algorithm gets you the shortest path cost from the initial node to each of the possible nodes, as well as the nodes you need to pass through in order to get to any other node
	graphs = [paths,dist,number_to_key,terminal_nodes]
	
	
	t_idx = np.argmin(np.array(dist)[terminal_nodes])
	goal_node = terminal_nodes[t_idx]
	
	min_dist = np.array(dist)[terminal_nodes]
	

	key_list = list(number_to_key.keys())
	val_list = list(number_to_key.values())
	 


	node_path = []
	node = goal_node
	#print("Path", ce)
	while node > 0: #we build the actual shortest path based on the cost metrics
		position = val_list.index(node)
		node_path.insert(0,key_list[position])
		node = paths[node]
	
	#print(node_path)
	
	return graphs,node_path

def sorted_k_partitions(seq, k):
	"""Returns a list of all unique k-partitions of `seq`.

	Each partition is a list of parts, and each part is a tuple.

	The parts in each individual partition will be sorted in shortlex
	order (i.e., by length first, then lexicographically).

	The overall list of partitions will then be sorted by the length
	of their first part, the length of their second part, ...,
	the length of their last part, and then lexicographically.
	"""
	n = len(seq)
	groups = []  # a list of lists, currently empty

	def generate_partitions(i):
		if i >= n:
	    		yield list(map(tuple, groups))
		else:
			if n - i > k - len(groups):
				for group in groups:
					group.append(seq[i])
					yield from generate_partitions(i + 1)
					group.pop()

			if len(groups) < k:
				groups.append([seq[i]])
				yield from generate_partitions(i + 1)
				groups.pop()

	result = generate_partitions(0)

	# Sort the parts in each partition in shortlex order
	result = [sorted(ps, key = lambda p: (len(p), p)) for ps in result]
	# Sort partitions by the length of each part, then lexicographically.
	result = sorted(result, key = lambda ps: (*map(len, ps), ps))

	return result
	




parser = argparse.ArgumentParser(description='Select tracks based on AEs')
parser.add_argument('--camera', type=str, help='Camera number')
parser.add_argument('--take', type=str, help='Take number')


args = parser.parse_args()

cameras = ['0','1','2']

watchbox = {}

for c in cameras:
	watchbox[c] = {}



#Wathbox coordinates. We ignore watchbox 0 from camera 2 because it overlaps with other watchboxes. We need to figure out how to use it in later work.
#watchbox['2'][0] = [200,1,1919,1079]
watchbox['2'][1] = [213,274,772,772]
watchbox['2'][2] = [816,366,1200,725]
watchbox['2'][3] = [1294,290,1881,765]
watchbox['1'][0] = [413,474,1072,772]
watchbox['0'][0] = [1,1,1919,1079]


tracks_path = "tracks_weakly_supervised/"

take = args.take #"351"
camera = args.camera #"3"

take_folder = "take_" + take + "_" + camera + "_0"

domain_shift = "smoke"	
root_path = "Elements/results/using_gt/CE1_" + domain_shift + "_50_True_True_True/" + take + "/ce_output.txt"



cap = cv2.VideoCapture(tracks_path+take_folder+".mp4")

pixel_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
pixel_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))




take_folder += '/'

frames_files = os.listdir(tracks_path+take_folder)

frames_files.sort(key=sort_file)

frame_range = (int(frames_files[0][:-4]),int(frames_files[-1][:-4]))

track_dict = {}


"""
This loop is used to load the bounding box information obtained previously using yolo+image classifier+tracker. We load it based on track ids into the track_dict.
"""
for ffile in frames_files:
	fe = open(tracks_path+take_folder+ffile)
	
	fe_lines = fe.readlines()
	
	for line in fe_lines:
		strip_line = line.strip().split(" ")
		
		track_id = int(strip_line[5])
		
		frame_num = int(ffile[:-4])
		
		
		
		if track_id not in track_dict:
			track_dict[track_id] = {"frame":np.array([]),"bbox":np.array([]), "class":[]} #For each track, we save the frames were it is present, bboxes as well as predicted classes
			
		if frame_num in track_dict[track_id]["frame"]:
			continue
			
		
		        
		track_dict[track_id]["frame"] = np.append(track_dict[track_id]["frame"],frame_num)

		box_voc = pbx.convert_bbox((float(strip_line[1]),float(strip_line[2]),float(strip_line[3]),float(strip_line[4])), from_type="yolo", to_type="voc", image_size=(pixel_width,pixel_height))
		#track_dict[track_id]["bbox"] = np.append(track_dict[track_id]["bbox"], box_voc)
		if track_dict[track_id]["bbox"].size == 0:
			track_dict[track_id]["bbox"] = np.expand_dims(np.array(box_voc),axis=0)
		else:
			track_dict[track_id]["bbox"] = np.concatenate((track_dict[track_id]["bbox"] , np.expand_dims(np.array(box_voc),axis=0)),axis=0)
		track_dict[track_id]["class"].append(int(strip_line[0]))
			
	
camera = str(int(camera)-1)

tolerance = 5

"""
aes format is the following {ground_truth track id:[[camera_id,watchbox,enter_frame,exit_frame], ...], ...}
If there is no exit frame registered, a -1 is put in the exit_frame
"""
aes = process_ae_file(root_path) #We load the file with the user annotations in terms of vicinal events. 
#pdb.set_trace()

#frame_aes = [{w:{"count":0,"tracks":[]} for w in watchbox[camera].keys()} for r in range(num_frames)]


print(aes)




ae_ranges = {}
ae_results = {}
min_frame = -1 #This is just to get the first frame of the entire video that is associated with a vicinal event. It is stored in min_frame
for ce in aes.keys():

	events_array = []
	for events in aes[ce]:
		if events[0] == camera:
			events_array.append(events[1])
			
			if min_frame == -1 or min_frame > events[2]:
				min_frame = events[2]
			
	print(ce, events_array)
	
#ae_ranges is just a structure used to get the ranges at which every ground truth track is present in every watchbox	
for ce in aes.keys():
	for events in aes[ce]:
		if events[0] == camera:
			if ce not in ae_ranges:
				ae_ranges[ce] = {}
				ae_results[ce] = {}
			
			end_range = num_frames+min_frame
			if events[3] > 0:
				end_range = events[3]
			
			ae_ranges[ce][events[1]]= np.array(list(range(events[2],end_range+1)))
			ae_results[ce][events[1]] = {} 

#This loop is used to associated tracks with each watchbox and get a matching score, and store al lthis information under ae_results
for t_key in track_dict.keys():

	track_dict[t_key]["unique_class"] = np.bincount(track_dict[t_key]["class"]).argmax() #This is the class that appears the most through all frames for track t_key, we will use it 
	for ce in ae_ranges.keys():
		for evt in ae_ranges[ce].keys():
			t_bool = np.in1d(track_dict[t_key]["frame"],ae_ranges[ce][evt])
			#t_indices = track_dict[track_id]["frame"][t_bool]
			#pdb.set_trace()

			if t_key not in ae_results[ce][evt]:
				ae_results[ce][evt][t_key] = {"score": 0, "frame":[], "bbox":[]}

			points = FindPoints(watchbox[camera][evt],track_dict[t_key]["bbox"][t_bool]) #Check if the location of track t_key is within the watchbox during frames t_bool
			
			ae_results[ce][evt][t_key]["score"] = sum(points)/len(ae_ranges[ce][evt]) #Matching score with the watchbox for ground truth track ce
			

			ae_results[ce][evt][t_key]["class"] = -1

			
			if any(points): #If there is overlap between detected track t_key and ground truth track ce
			
				ae_results[ce][evt][t_key]["class"] = np.bincount(np.array(track_dict[t_key]["class"])[t_bool]).argmax()
				try:
					first_bbox = track_dict[t_key]["bbox"][t_bool][points][0]
				except:
					pdb.set_trace()
				last_bbox = track_dict[t_key]["bbox"][t_bool][points][-1]
				
				ae_results[ce][evt][t_key]["bbox"] = [first_bbox,last_bbox]
				
				first_frame= track_dict[t_key]["frame"][t_bool][points][0]
				last_frame = track_dict[t_key]["frame"][t_bool][points][-1]
				ae_results[ce][evt][t_key]["frame"] = [first_frame,last_frame]
			
			

				
			
#print(ae_results)

graph_links = {}

graph_nodes = {}


number_vertices = {}

#Forward pass: we are gonig to establish edges betwen non overlapping subtracks

for ce in ae_results.keys(): #For each ground truth track ce
	
	graph_links[ce] = {}
	graph_nodes[ce] = {}
	number_vertices[ce] = 1
	
	if int(ce) < 4:
		ae_class = 1
	else:
		ae_class = 0
	
	past_evts = []		
	for evt in ae_results[ce].keys(): #For each watchbox evt
	
		past_evts.insert(0,evt)

		key_val = np.array([[t_key,ae_results[ce][evt][t_key]["score"], ae_results[ce][evt][t_key]["frame"], ae_results[ce][evt][t_key]["bbox"]] for t_key in ae_results[ce][evt].keys() if track_dict[t_key]["unique_class"] == ae_class], dtype=object) #Alternative is checking ae_results[ce][evt][t_key]["class"]
		
		if key_val.size == 0:
			graph_nodes[ce][evt] = []
			continue
		
		#ind = np.argpartition(key_val[:,1], -5)[-5:]
		#ind = np.flip(ind[np.argsort(key_val[ind,1])])
		
		try:
			ind = np.flip(np.argsort(key_val[:,1])) #Create descending order of tracks based on how well they match gt track ce during its stay in watchbox evt. Key val already filters according to classes
		except:
			pdb.set_trace()

		graph_nodes[ce][evt] = key_val[ind] #We put the information about tracks inside graph_nodes
		
		links = {}
		
		
			
		if evt not in graph_links[ce]:
			graph_links[ce][evt] = {"edge":{},"bbox":{},"time":{}, "score":{}, "ae":{}, "final":{}, "self_score":{}, "group_score":{}}
			
		if len(past_evts) == 1:
			graph_links[ce][-1] = {"edge":{0: []}, "score":{0: []}, "ae":{0: []}, "final":{0:[]}, "self_score":{0:1.0}, "group_score":{0:-1}}
			
			
			
			
			
		#In this part we try to establish the links between nodes
		for pevt in past_evts: #Iterates over all watchboxes in a sequence
			for kv in graph_nodes[ce][pevt]: #kv goes through all watchboxes 
				if kv[1]:
						
							
						
					for kv2 in graph_nodes[ce][evt]: #kv2 is the data in the last watchbox in sequence
						if kv2[1]:
							if kv2[0] not in graph_links[ce][evt]["edge"]: #Initialize node as part of an edge to kv2
								graph_links[ce][evt]["edge"][kv2[0]] = []
								graph_links[ce][evt]["bbox"][kv2[0]] = []
								graph_links[ce][evt]["time"][kv2[0]] = []
								graph_links[ce][evt]["score"][kv2[0]] = []
								graph_links[ce][evt]["self_score"][kv2[0]] = kv2[1] #Transfer matching score 
								graph_links[ce][evt]["group_score"][kv2[0]] = -1
								graph_links[ce][evt]["ae"][kv2[0]] = []
								graph_links[ce][evt]["final"][kv2[0]] = []
								number_vertices[ce] += 1
								
							if kv2[0] not in graph_links[ce][-1]["edge"][0]: #Initialize data for sink node connected to first watchbox
								graph_links[ce][-1]["edge"][0].append(kv2[0])
								graph_links[ce][-1]["score"][0].append(kv2[1])
								graph_links[ce][-1]["ae"][0].append(evt)
								graph_links[ce][-1]["group_score"][0] = -1
								graph_links[ce][-1]["final"][0] = []
								graph_links[ce][-1]["self_score"][0] = 1.0
								
								"""
								if evt + 1 <= 1: #If the difference between watchbox events in sequence is lower than 1
									distance_cost = 0
								else:
									distance_cost = evt
								
								graph_links[ce][-1]["final"][0].append((distance_cost)+(1-kv2[1])) #Cost measurement is based on how distant in terms of the watchbox sequence is it plus the inverse of the similarity score with the ae
								"""
								
							if kv[0] != kv2[0] or pevt != evt: #This will evaluate false only when we are in the same watchbox and we are evaluating the same track_id
							
								#if kv2[0] == 14 and kv[0] == 264 and ce == 0 and evt == 1:
								#	pdb.set_trace()
							
								if kv2[2] and kv[2] and kv2[2][0] > kv[2][1] and not (kv2[2][0] in track_dict[kv[0]]["frame"] and kv[0] != kv2[0]): #Frames shouldn't overlap
									graph_links[ce][pevt]["edge"][kv[0]].append(kv2[0]) #add edges from current node to other nodes
									graph_links[ce][pevt]["bbox"][kv[0]].append(np.linalg.norm(kv[3][1][:2]-kv2[3][0][:2]))
									graph_links[ce][pevt]["time"][kv[0]].append(kv2[2][0]-kv[2][1])
									
							
										
									#Not used
									if kv[0] == kv2[0] and past_evts.index(pevt) == len(past_evts)-2: #If the next track is the same we should go to that one (lower cost to 0)
										graph_links[ce][pevt]["score"][kv[0]].append(1.0)
									else:
										graph_links[ce][pevt]["score"][kv[0]].append(kv2[1])
										
									
										
									
									
									graph_links[ce][pevt]["ae"][kv[0]].append(evt)
									graph_links[ce][pevt]["final"][kv[0]].append(0)
									

									"""
									if evt != pevt: #if kv is not from the same watchbox as kv2
										
										if evt-pevt == 1 and kv[0] == kv2[0]:
											cost = 0
										
										else:
											if evt-pevt <= 1:
												distance_cost = 0
											else:
												distance_cost = evt-pevt-1
												
											cost = (1-graph_links[ce][pevt]["group_score"][kv[0]]) + (1-graph_links[ce][evt]["group_score"][kv2[0]]) + distance_cost
									else:
										cost = 0
										
									graph_links[ce][pevt]["final"][kv[0]].append(cost)
									"""
			

			"""			
			if evt == pevt:	#if we are analyzing nodes from the same watchbox
				for edge in graph_links[ce][pevt]["edge"].keys():
					graph_links[ce][evt]["group_score"][edge] = recursive_search(graph_links[ce][pevt], edge) #We will try to propagate the self_scores that tell us the paths with the most similarity in order to create a kind of opportunity cost
				for edge in graph_links[ce][pevt]["edge"].keys():
					for e_idx,e in enumerate(graph_links[ce][pevt]["edge"][edge]):
								
						
					
						graph_links[ce][pevt]["final"][edge][e_idx] = graph_links[ce][evt]["group_score"][edge] - (graph_links[ce][evt]["self_score"][edge] + graph_links[ce][evt]["group_score"][e])
			
			"""
				
			

			
		
		
		"""
		max_score = 0
		max_key = -1
		for t_key in ae_results[ce][evt].keys():
		
			#detected_class = np.bincount(track_dict[t_key]["class"]).argmax()
			
			if detected_class == ae_class:
			
				res = ae_results[ce][evt][t_key]
				
				if res > max_score:
					max_score = res
					max_key = t_key
		"""
		#print(key_val[ind,0], "matches with track", ce, "watchbox", evt, "score", key_val[ind,1], "frames", key_val[ind,2], "bboxes", key_val[ind,3])
		


#Backwards pass: compute the cost between each link. "self_score" is used not "score". "final" is where the cost is saveed
for ce in graph_links.keys():
	max_evt = max(graph_links[ce].keys())
	evts = sorted(list(graph_links[ce].keys()), reverse=True)
	for evt in evts: #We start in the last watchbox in a sequence and finish in the first

		for edge in graph_links[ce][evt]["edge"].keys():
			graph_links[ce][evt]["group_score"][edge] = recursive_search2(graph_links[ce], evt, edge, max_evt-evt) #We will try to propagate the self_scores that tell us the paths with the most similarity in order to create a kind of opportunity cost. "group_score" saves the greatest matching score associated with any given path at this watchbox
				
		for edge in graph_links[ce][evt]["edge"].keys():
			for e_idx,e in enumerate(graph_links[ce][evt]["edge"][edge]):
			
				cevt = graph_links[ce][evt]["ae"][edge][e_idx]
			
				cost = graph_links[ce][evt]["group_score"][edge] - (graph_links[ce][evt]["self_score"][edge] + graph_links[ce][cevt]["group_score"][e]) #The cost is calculated as the best score you can achieve in this watchbox starting at this node by following the optimum path minus the penalty over taking the edge that doesn't result in the best path
				
				if graph_links[ce][evt]["ae"][edge][e_idx] - evt == 1 and e == edge: #If the next subtrack is part of the same track we should go to that one (lower cost to 0)
					cost = 0
				elif graph_links[ce][evt]["ae"][edge][e_idx] - evt > 1: #There is a penalty for skipping through watchboxes in a sequence
					cost += graph_links[ce][evt]["ae"][edge][e_idx] - evt - 1
						
						
				if evt == -1:
					graph_links[ce][evt]["final"][edge].append(0)
				try:
					graph_links[ce][evt]["final"][edge][e_idx] = cost
				except:
					pdb.set_trace()
					print("problem")
	


#Topological sort algorithm	
class Graph:
	def __init__(self,vertices):

		self.V = vertices # No. of vertices

		# dictionary containing adjacency List
		self.graph = defaultdict(list)

		# function to add an edge to graph
	def addEdge(self,u,v,w):
		self.graph[u].append((v,w))


		# A recursive function used by shortestPath
	def topologicalSortUtil(self,v,visited,stack):

		# Mark the current node as visited.
		visited[v] = True

		# Recur for all the vertices adjacent to this vertex
		if v in self.graph.keys():
			for node,weight in self.graph[v]:
				if visited[node] == False:
    					self.topologicalSortUtil(node,visited,stack)

		# Push current vertex to stack which stores topological sort
		stack.append(v)


		''' The function to find shortest paths from given vertex.
		It uses recursive topologicalSortUtil() to get topological
		sorting of given graph.'''
	def shortestPath(self, s):

		# Mark all the vertices as not visited
		visited = [False]*self.V
		stack =[]

		# Call the recursive helper function to store Topological
		# Sort starting from source vertices
		for i in range(self.V):
			if visited[i] == False:
				self.topologicalSortUtil(s,visited,stack)


		# Initialize distances to all vertices as infinite and
		# distance to source as 0
		dist = [float("Inf")] * (self.V)
		
		dist[s] = 0
		
		# Process vertices in topological order
		#print(stack[::-1])
		
		node_list = []
		
		paths = {v:-1 for v in range(self.V)}
		
		idx = 0
		
		while stack:

			# Get the next vertex from topological order
			i = stack.pop()
			
			#pdb.set_trace()
			# Update distances of all adjacent vertices
			for node,weight in self.graph[i]:
				if dist[node] > dist[i] + weight:
					dist[node] = dist[i] + weight
					paths[node] = i
					
			idx += 1
			
		#print(paths)

		# Print the calculated shortest distances
		#for i in range(self.V):
		#	print ((dist[i]) if dist[i] != float("Inf") else  "Inf" ,end=" ")
			
		return paths,dist


graphs = {}
node_path = {}

#Get shortest path initially for each ground truth track graph
for ce in graph_links.keys():


	graph,node_p = get_shortest_path(graph_links[ce], number_vertices[ce])
	
	graphs[ce] = graph
	node_path[ce] = node_p
	

	


print(node_path)

#Iterate until there are no more conflicting shortest paths
#TODO what happens if there are no more nodes available
while True:

	try:
		key_val = np.array([[ce,graphs[ce][1][graphs[ce][2][node_path[ce][-1]]]] for ce in graph_links.keys()]) #Get the final cost per path
	except:
		pdb.set_trace()
	ind = np.argsort(key_val[:,1]) #We sort the shortest path costs in ascending order (from lower to higher)

	nodes_so_far = []
	repeated_nodes = {}
	all_nodes = {}
	for k_idx in ind: #We save the nodes of the least cost shortest paths, and save the repeated nodes in the higher cost shortest paths
		ce = int(key_val[k_idx][0])
		
		repeated_nodes_flag = False
		temp_nodes_so_far = []
		
		for node in node_path[ce]:
		
				
			if node not in nodes_so_far:
				temp_nodes_so_far.append(node)
				
			else:
				repeated_nodes_flag = True
				
				
				
		
				
		if not repeated_nodes_flag:
			nodes_so_far.extend(temp_nodes_so_far)
		else:

			all_nodes = [str(track)+"_"+str(evt) for evt in graph_links[ce].keys() for track in graph_links[ce][evt]["final"].keys()]
			repeated_nodes[ce] = list(set(nodes_so_far) & set(all_nodes))



				
		

	if not repeated_nodes:
		break

	for ce in repeated_nodes.keys(): #If there are repetead nodes, calculate the shortest path without these nodes for all the previous shortest paths that relied on those

		
		graph,node_p = get_shortest_path(graph_links[ce], number_vertices[ce], repeated_nodes[ce])
		
		
		graphs[ce] = graph
		node_path[ce] = node_p
		
	#print(repeated_nodes)
	#print(node_path)


print(node_path)

#Get all the tracks that we should extract based on the previous shortest paths (right now we are ignoring that a track could have subtracks that are part of different gt tracks)

tracks_to_keep = []

for ce in node_path.keys():
	for node in node_path[ce]:
		track = int(node.split("_")[0])
		if track not in tracks_to_keep:
			tracks_to_keep.append(track)
			
			
print(tracks_to_keep)

#Save ther results in yolo format

write_dir = "selection_files/" + args.take + "_" + args.camera + "/"
if os.path.exists(write_dir):
	os.system("rm -r " + write_dir)    
os.makedirs(write_dir)

for frame in range(frame_range[0],frame_range[1]+1):
	txt_line = ""
	for track_id in tracks_to_keep:
		if frame in track_dict[track_id]["frame"]:

			frame_idx = np.where(track_dict[track_id]["frame"] == frame)[0][0]
			yolo_box = pbx.convert_bbox(tuple(track_dict[track_id]["bbox"][frame_idx]), from_type="voc", to_type="yolo", image_size=(pixel_width,pixel_height))
			txt_line += "%d %f %f %f %f\n" % (track_dict[track_id]["unique_class"], *yolo_box)
			
	if txt_line:
		f_txt = open(write_dir+str(frame)+".txt", "w")
		f_txt.write(txt_line)
		f_txt.close()

#pdb.set_trace()

'''
common_elements = []

ces = list(combinations(graph_links.keys(), 2))


global_intersection = set([])
combinable_items = []
combinable_ces = []

for comb in ces: #Check if there are common nodes visited
	intersected = list(set(node_path[comb[0]]) & set(node_path[comb[1]]))
	
	"""
	if intersected:
		for i in intersected:
			for c in comb:
				assignment = [c,i]
				if assignment not in combinable_items
					combinable_items.append(assignment)
	"""
	
	if intersected:
		for c in comb:
			if c not in combinable_ces:
				combinable_ces.append(c)
	
	#common_elements.append(intersected)
	global_intersection = global_intersection.union(intersected)
	


l_global_intersection = list(global_intersection)

k = len(combinable_ces)

#possible_combinations = sorted_k_partitions(l_global_intersection,len(combinable_ces))

possible_combinations = []

for i in (range(k)):
	for groups in sorted_k_partitions(l_global_intersection, k-i):
		for perm in permutations(groups+[tuple() for j in range(i)]):
			possible_combinations.append(perm)

#possible_combinations.extend([[(),tuple(l_global_intersection)]])
print(possible_combinations)


total_cost = []

node_path_comb = []

#pdb.set_trace()
non_combinable_ces = list(set(graph_links.keys()) - set(combinable_ces))
non_combinable_nodes = []
for nc in non_combinable_ces:
	non_combinable_nodes.extend(node_path[nc])
	
non_combinable_nodes = set(non_combinable_nodes)


for pc in possible_combinations:
	combination_cost = 0
	node_path_comb.append([])
	if len(pc) != len(combinable_ces):
		total_cost.append(10000) #Really large value
		continue
	for ce_idx,ce in enumerate(combinable_ces):
		allowed = set(pc[ce_idx])
		disallowed = list((global_intersection | non_combinable_nodes) - allowed)

		disallowed_list = [d for d in disallowed if int(d.split('_')[1]) in graph_links[ce].keys() and int(d.split('_')[0]) in graph_links[ce][int(d.split('_')[1])]["edge"].keys()]


		graph,node_p = get_shortest_path(graph_links[ce], number_vertices[ce], disallowed_list)

		node_path_comb[-1].append(node_p)
		combination_cost += graph[1][graph[2][node_p[-1]]] #Get distance cost of last node
	total_cost.append(combination_cost)
	
best_combination_idx = np.argmin(total_cost)
print("Best paths", total_cost[best_combination_idx], possible_combinations[best_combination_idx], node_path_comb[best_combination_idx])

for ce in node_path.keys():
	if ce not in combinable_ces:
		print(ce,node_path[ce])
	else:
		ce_idx = combinable_ces.index(ce)
		print(ce,node_path_comb[best_combination_idx][ce_idx])

"""

tests = list(product(*combinable_items))

for t in tests:
	for ce in t:
		to_remove = global_intersection - set([ce[1]])
		graph,node_p = get_shortest_path(graph_links[ce], number_vertices[ce])
	

print(tests)
		

"""	


print(graphs)



pdb.set_trace()
'''
"""
	
for f_idx in range(len(frame_aes)):
	for ce in aes.keys():
		for events in aes[ce]:
			if events[0] == camera:
				real_frame_idx = f_idx+min_frame
				if real_frame_idx > events[2] and (real_frame_idx < events[3] or events[3] < 0):
					frame_aes[f_idx][events[1]]["count"] += 1
					
				for t_key in track_dict.keys():
					if real_frame_idx in track_dict[t_key]["frame"]:
						rf_idx = track_dict[t_key]["frame"].index(real_frame_idx)
						if FindPoint(watchbox[camera][events[1]],track_dict[t_key]["bbox"][rf_idx]):
							frame_aes[f_idx][events[1]]["tracks"].append(t_key)
		
							

print(frame_aes)
"""
"""

for t_key in track_dict.keys():
	for ce in aes.keys():
		
		#if (t_key == 3 or t_key == 13 or t_key == 14) and (ce == 2 or ce == 3):
		#	pdb.set_trace()
		
		check = []
		for events in aes[ce]:
			if events[0] == camera:
			
				
			
				try:
					b_array = np.where((np.array(track_dict[t_key]["frame"]) <= events[2]+tolerance) & (np.array(track_dict[t_key]["frame"]) >= events[2]-tolerance))[0]
				except:
					pdb.set_trace()
				
				if (t_key == 3 or t_key == 13 or t_key == 14) and (ce == 2 or ce == 3):
					print("Entrance", " Watchbox: ", events[1], " Detected Track: ", t_key, " Original Track: ", ce, np.array(track_dict[t_key]["frame"])[b_array], " Reference: ", events[2])
				
				if b_array.size > 0:
					b_idx = b_array[0]

					bboxes = np.array(track_dict[t_key]["bbox"])[b_array][:2]
					
					if len(bboxes.shape) == 1:
						bboxes = np.expand_dims(bboxes,0)
					
					check.append(any(FindPoints(watchbox[camera][events[1]],bboxes)))
				else:
					check.append(False)
				
				if events[3] > -1:

					b_array = np.where((np.array(track_dict[t_key]["frame"]) <= events[3]+tolerance) & (np.array(track_dict[t_key]["frame"]) >= events[3]-tolerance))[0]
					
					if (t_key == 3 or t_key == 13 or t_key == 14) and (ce == 2 or ce == 3):
						print("Exit", " Watchbox: ", events[1], " Detected Track: ", t_key, " Original Track: ", ce, np.array(track_dict[t_key]["frame"])[b_array], " Reference: ", events[3])
					
					if b_array.size > 0:
						b_idx = b_array[0]

						bboxes = np.array(track_dict[t_key]["bbox"])[b_array][:2]
						if len(bboxes.shape) == 1:
							bboxes = np.expand_dims(bboxes,0)
						check[-1] = check[-1] and (not all(FindPoints(watchbox[camera][events[1]],bboxes)))
					else:
						check[-1] = False

		detected_class = np.bincount(track_dict[t_key]["class"]).argmax()
		
		if int(ce) < 4:
			ae_class = 1
		else:
			ae_class = 0
			
		if check and any(check):
			txt_str = "Class: "
			if detected_class == ae_class:
				txt_str += "True"
				print(t_key, "matches", ce, check, txt_str)
			else:
				txt_str += "False"
			
			
				
					
					
"""

	
					
				
			

