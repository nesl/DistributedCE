#!/usr/bin/python
# -*- coding: UTF-8 -*-

import zmq
import argparse
import threading

def peer_run(ctx):
    """ this is the run method of the PAIR thread that logs the messages
    going through the broker """
    sock = ctx.socket(zmq.PAIR)
    sock.connect("inproc://peer")  # connect to the caller
    sock.send(b"")  # signal the caller that we are ready
    while True:
        try:
            topic = sock.recv_string()
            obj = sock.recv_pyobj()
        except Exception:
            topic = None
            obj = sock.recv()
        print(f"\n !!! peer_run captured message with topic {topic}, obj {obj}. !!!\n")

def main():

    parser = argparse.ArgumentParser(description='Broker')

    parser.add_argument('--port_consumers', type=int, help='Port facing consumers')
    parser.add_argument('--port_producers', type=int, help='Port facing produces')
    args = parser.parse_args()


    context = zmq.Context()
    
    """
    cap = context.socket(zmq.PAIR)
    cap.bind("inproc://peer")
    cap_th = threading.Thread(target=peer_run, args=(context,), daemon=True)
    cap_th.start()
    cap.recv()  # wait for signal from peer thread
    print("cap received message from peer, proceeding.")
    """

    # Socket facing producers
    frontend = context.socket(zmq.XPUB)
    frontend.bind("tcp://*:" + str(args.port_producers))
    print("producers: ", "tcp://*:" + str(args.port_producers))
    # Socket facing consumers
    backend = context.socket(zmq.XSUB)
    backend.bind("tcp://*:" + str(args.port_consumers))
    print("consumers: ", "tcp://*:" + str(args.port_consumers))
    print("Setting proxy...")
    zmq.proxy(frontend, backend)

    # We never get here…
    frontend.close()
    backend.close()
    context.term()

if __name__ == "__main__":
    main()
