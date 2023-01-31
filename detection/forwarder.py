import socket 
import numpy as np
import cv2
import pdb
import argparse

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise RuntimeError("socket connection broken")
        data.extend(packet)
    return data

def mysend(socket, msg, MSGLEN):
    totalsent = 0
    while totalsent < MSGLEN:
        sent = socket.send(msg[totalsent:])
        if sent == 0:
            raise RuntimeError("socket connection broken")
        totalsent = totalsent + sent
        
        	    
parser = argparse.ArgumentParser(description='Forwarder.')
parser.add_argument('--address_source', type=str, help='Source address of camera')
parser.add_argument('--address_sink', type=str, help='Send images to this address')
parser.add_argument('--port_source', type=int, help='Source port of camera')
parser.add_argument('--port_sink', type=int, help='Send images to this port')

args = parser.parse_args()

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.settimeout(5)
client_socket.connect((args.address_source, args.port_source))

client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket2.connect((args.address_sink, args.port_sink))
imgsz = 800*600*4
while True:

    #imgidx = client_socket.recv(2)
    
    
    #imgsz = int(imgsz.decode("utf8"))
    try:
        imgidx = recvall(client_socket, 2)
        print(imgidx)
        image =recvall(client_socket, imgsz)
    except Exception as e:
        print('Timeout', e)
        #client_socket.connect((args.address_source, args.port_source))
        continue
        
    try:
        mysend(client_socket2,imgidx,len(imgidx))

        
        mysend(client_socket2,image,len(image))

        #image = client_socket.recv(imgsz)
    except Exception as e:
        print('Timeout 2', e)
        #client_socket2.connect((args.address_sink, args.port_sink))
        continue

    
    
