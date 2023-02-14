#!/usr/bin/python
# File: emptynet.py
from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import pdb
import time

import os
import signal
from signal import SIGKILL
import subprocess


import csv

# Read in all our lines and get the data structure
def read_data(points_filepath="locations.txt"):

    points_data = []

    # Open the points file where we store all our waypoints/camera positions/etc
    with open(points_filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            row = [x.strip() for x in row]
            if not row:
                continue
            # We ignore rows with '#' comments
            comments = [1 if "#" in x else 0 for x in row]
            if comments[0] == 1:  # The whole row is a comment, and we ignore
                continue
            elif 1 in comments: # There is a comment at a later point
                comment_index = comments.index(1)
                last_obj = row[comment_index].split("#")[0]
                row = row[:comment_index]
                row.append(last_obj.strip())

            # Do some basic error checking
            assert len(row) == 7

            #  Appent our data points
            desc = row[-1]
            row = [float(x) for x in row[:-1]]
            points_data.append(row + [desc])

    return points_data
    
    

def emptyNet():

	# First, read in our different sensor locations
	points_data = read_data("../simulator/locations.txt")

	"Create an empty network and add nodes to it."

	net = Mininet( controller=Controller )

	info( '*** Adding controller\n' )
	net.addController( 'c0' )

	info( '*** Adding hosts\n' )
	
	simulator_ip_address = '10.0.0.1'
	central_server_ip_address = '10.0.0.2'
	broker_1_ip_address = '10.0.0.3'
	broker_2_ip_address = '10.0.0.4'
	
	h1 = net.addHost( 'h1', ip=simulator_ip_address )
	h2 = net.addHost( 'h2', ip=central_server_ip_address )
	h3 = net.addHost( 'h3', ip=broker_1_ip_address )
	h4 = net.addHost( 'h4', ip=broker_2_ip_address )

	#h2 = net.addHost( 'h2', ip='10.0.0.2' )
	#h3 = net.addHost( 'h3', ip='10.0.0.3' )
	#h4 = net.addHost( 'h4', ip='10.0.0.4' )
	#h5 = net.addHost( 'h5', ip='10.0.0.5' )


	info( '*** Adding switch\n' )
	s1 = net.addSwitch( 's1' )

	info( '*** Creating links\n' )

	client_hosts = []
	client_data = []
	host_id = 5

	load_port = 55001
	broker_1_port_consumers = 54001
	broker_1_port_producers = 54002
	broker_2_port_consumers = 54003
	broker_2_port_producers = 54004
	#load_port = 58001
	
	
	

	# Add our hosts
	for camera_location in points_data:
	
		hostname_loader = "h" + str(host_id)
		loader_ip_address = "10.0.0." + str(host_id)
		client_hosts.append( net.addHost(hostname_loader, ip=loader_ip_address ) )

		loader_arguments = [ simulator_ip_address, str(load_port), broker_1_ip_address, str(broker_1_port_consumers), broker_2_ip_address, str(broker_2_port_producers), str(host_id-4)]
		print(loader_arguments)
		#host_id += 1

		"""
		hostname_forwarder = "h" + str(host_id)
		forwarder_ip_address = "10.0.0." + str(host_id)
		client_hosts.append( net.addHost(hostname_forwarder, ip=forwarder_ip_address) )

		forwarder_arguments = [ '10.0.0.1', loader_ip_address, str(forward_port), str(load_port) ]
		print(forwarder_arguments)
		"""
		
		client_data.append(loader_arguments)
		#client_data.append(forwarder_arguments)

		host_id += 1
		load_port+=1
		#forward_port+=1
    	
    
	# Add network links
	client_links = []
	net.addLink(h1, s1)
	net.addLink(h2, s1)
	net.addLink(h3, s1)
	net.addLink(h4, s1)
	for host in client_hosts:
		print(host)
		link = net.addLink(host, s1)
		client_links.append(link)
		
	# Traffic camera links between loader and switch
	#link_tc1 = net.linksBetween(client_hosts[0], s1)
	#link_tc2 = net.linksBetween(client_hosts[2], s1)
	
	pids_to_kill = []


	info( '*** Starting network\n')
	net.start()

	# First, load up CARLA and the xterm for running the connection
	h1.cmd("./h1_1.sh")
	time.sleep(5)
	h1_pid = h1.cmd("xterm -hold -e './h1_2.sh " + simulator_ip_address + "' &")
	
	#ZMQ brokers
	h3_pid = h3.cmd("xterm -hold -e './broker.sh " + str(broker_1_port_consumers) + " " + str(broker_1_port_producers) + "' &")
	h4_pid = h4.cmd("xterm -hold -e './broker.sh " + str(broker_2_port_consumers) + " " + str(broker_2_port_producers) + "' &")
	

	# Next, set up all of our loaders, which must perform YOLO

	# Perform recognition

	time.sleep(4)
	for i in range(0, len(client_hosts)):
		command = "xterm -hold -e './h2.sh " + ' '.join(client_data[i]) + "' &"
		client_hosts[i].cmd(command)

	# h2_pid = h2.cmd("xterm -hold -e './h2.sh' &")
	# h4_pid = h4.cmd("xterm -hold -e './h4.sh' &")

	# Then, load up our forwarders
	time.sleep(5)

	#Central server
	h2_pid = h2.cmd("xterm -hold -e './central_server.sh " + broker_1_ip_address + " " + str(broker_1_port_producers) + " " + broker_2_ip_address + " " + str(broker_2_port_consumers) + "' &")
	
	#Wait for central server to load
	time.sleep(5)
	"""
	for i in range(1, len(client_hosts), 2):
		command = "xterm -hold -e './h3.sh " + ' '.join(client_data[i]) + "' &"
		client_hosts[i].cmd(command)
	"""

	# h3_pid = h3.cmd("xterm -hold -e './h3.sh' &")
	# h5_pid = h5.cmd("xterm -hold -e './h5.sh' &")

	print(h1_pid,h2_pid,h3_pid,h4_pid)

	# Add all our pids to be killed upon exit
	carla_pid = subprocess.check_output("ps aux | grep CarlaUE4-Linux-Shipping", shell=True)
	pids_to_kill.append(carla_pid.decode().split()[1])
	
	pids_to_kill.append(h1_pid.split()[-1])

	#pids_to_kill.append(h3_pid.split()[-1])
	#pids_to_kill.append(h5_pid.split()[-1])

	#pids_to_kill.append(h2_pid.split()[-1])
	#pids_to_kill.append(h4_pid.split()[-1])


	#h1.cmd('su nesl -c simulator/CARLA_0.9.10/CarlaUE4.sh &') #;  source venv/bin/activate; cd simulator/; scenic pedestrian.scenic --simulate')
	#h1.cmd('source venv/bin/activate')
	#h1.cmd('cd simulator/')
	#h1.cmd('scenic pedestrian.scenic --simulate')
	#h2.cmd('cd detection/')
	#h2.cmd('source yolo/bin/activate')
	#h2.cmd('python load.py')
	#h3.cmd('source venv/bin/activate')
	#h3.cmd('cd detection/')
	#h3.cmd('python forwarder.py')
	info( '*** Running CLI\n' )
	CLI( net )

	info( '*** Stopping network' )
	net.stop()

	# Kill the PIDs
	for pid_x in pids_to_kill:
		os.kill(int(pid_x), signal.SIGKILL)
    
    
    	

if __name__ == '__main__':
    setLogLevel( 'info' )
    emptyNet()
