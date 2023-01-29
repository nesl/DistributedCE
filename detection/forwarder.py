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
            return None
        data.extend(packet)
    return data


parser = argparse.ArgumentParser(description='Forwarder.')
parser.add_argument('--address_source', type=str, help='Source address of camera')
parser.add_argument('--address_sink', type=str, help='Send images to this address')
parser.add_argument('--port_source', type=int, help='Source port of camera')
parser.add_argument('--port_sink', type=int, help='Send images to this port')

args = parser.parse_args()

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((args.address_source, args.port_source))

client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket2.connect((args.address_sink, args.port_sink))
imgsz = 800*600*4
while True:

    #imgidx = client_socket.recv(2)
    
    
    #imgsz = int(imgsz.decode("utf8"))
    imgidx = recvall(client_socket, 2)
    client_socket2.sendall(imgidx)
    image =recvall(client_socket, imgsz)
    client_socket2.sendall(image)
    #image = client_socket.recv(imgsz)
    

    
    
