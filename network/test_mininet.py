#!/usr/bin/python
# File: emptynet.py
from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import pdb


def emptyNet():

    "Create an empty network and add nodes to it."

    net = Mininet( controller=Controller )

    info( '*** Adding controller\n' )
    net.addController( 'c0' )

    info( '*** Adding hosts\n' )
    h1 = net.addHost( 'h1', ip='10.0.0.1' )
    h2 = net.addHost( 'h2', ip='10.0.0.2' )
    h3 = net.addHost( 'h3', ip='10.0.0.3' )


    info( '*** Adding switch\n' )
    s1 = net.addSwitch( 's1' )

    info( '*** Creating links\n' )
    net.addLink( h1, s1 )
    net.addLink( h2, s1 )
    net.addLink( h3, s1 )


    info( '*** Starting network\n')
    net.start()


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

if __name__ == '__main__':
    setLogLevel( 'info' )
    emptyNet()
