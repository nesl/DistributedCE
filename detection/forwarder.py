import socket 
import numpy as np
import cv2
import pdb


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data



client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('10.0.0.1', 55001))

client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket2.connect(('10.0.0.2', 58000))
imgsz = 800*600*4
while True:

    #imgidx = client_socket.recv(2)
    
    
    #imgsz = int(imgsz.decode("utf8"))
    imgidx = recvall(client_socket, 2)
    client_socket2.sendall(imgidx)
    image =recvall(client_socket, imgsz)
    client_socket2.sendall(image)
    #image = client_socket.recv(imgsz)
    

    
    
