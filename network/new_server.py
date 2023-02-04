import pdb
import socket
import threading
import socketserver
import time
import pickle
import os
import zmq
import argparse
from data_format import Data, Message, Location


parser = argparse.ArgumentParser(description='Server')
parser.add_argument('--ip_from_sensors', type=str, help='Address to receive data from sensors')
parser.add_argument('--port_from_sensors', type=int, help='Port receving data from sensors')
parser.add_argument('--ip_to_sensors', type=str, help='Address to send data to sensors')
parser.add_argument('--port_to_sensors', type=int, help='Port sending data to sensors')
args = parser.parse_args()

ctx = zmq.Context()

address_from_sensors = "tcp://%s:%s" % (args.ip_from_sensors,str(args.port_from_sensors))
address_to_sensors = "tcp://%s:%s" % (args.ip_to_sensors,str(args.port_to_sensors))

print("address_from_sensors: ", address_from_sensors)
print("address_to_sensors: ", address_to_sensors)

num_queries = 0



cameras_ips = []
queries = []

'''
def get_query(data):


    data = InputStream(data.strip())

    # lexer
    lexer = stlLexer(data)
    stream = CommonTokenStream(lexer)
    # parser
    parser = ourGrammer(stream)
    tree = parser.expr()
    spatial_costs = [[0, 1],[1, 0]]
    spatial_representation = {"1": 0, "2":1, "3": 2, "4":3}
    # Hacky way - no time to fix NN so have to opt for manual ignoring bboxes

    visitor = ourVisitor(spatial_representation, spatial_costs, parser)

    return visitor,tree


def get_ordered_atomic_functions(visitor,tree):

    visitor.test = True
    visitor.visit(tree)
    visitor.test = False

    return visitor.function_order
'''



class Query():
    def __init__(self, atomic_functions, id):
        self.atomic_functions = atomic_functions
        #self.visitor = visitor
        #self.tree = tree
        self.cameras = {}
        self.state = {}
        self.id = id



cameras_sockets = []

"""
#This represent a single camera
class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):

    def __init__(self,request,client_address,server):
        super().__init__(request,client_address,server)
        self.state = {}

    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        cur_thread = threading.current_thread()
        response = bytes("{}: {}".format(cur_thread.name, data), 'ascii')
        self.request.sendall(response)


    def setup(self):
        print("Initiated connection with ", self.client_address)
        cameras_ips.append(self.client_address[0])
        cameras_sockets.append(self.request)
        self.state = {}
        first_functions = []
        for q in queries:
            self.state[q.id] = q.atomic_functions[0]
            first_functions.extend(q.atomic_functions[0])

        data = pickle.dumps(list(set(first_functions)))
        self.request.sendall(data)
    def finish(self) -> None:
        print("Ended connection with ", self.client_address)
        cameras_ips.remove(self.client_address[0])

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.allow_reuse_address = True
"""

#This handles our queries
class ThreadedUnixRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        data = str(self.request.recv(1024), 'ascii')
        print(data)
        visitor, tree = get_query(data)
        atomic_functions = get_ordered_atomic_functions(visitor, tree)
        query = Query(visitor, tree, atomic_functions, 0)
        queries.append(query)
        x = threading.Thread(target=thread_function, args=(query,), daemon=True)
        x.start()


class ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    def __init__(self, address, handler):
        super().__init__(address, handler)
        self.allow_reuse_address = True



def setup_thread(query, visitor, tree, query_num):
    sock_from_sensors = ctx.socket(zmq.SUB)
    sock_from_sensors.connect(address_from_sensors)
    sock_to_sensors = ctx.socket(zmq.PUB)
    sock_to_sensors.connect(address_to_sensors)
    #print(address_to_sensors)
    topics = query.atomic_functions[0]

    message_num = 0

    for t in topics:
        sock_from_sensors.setsockopt(zmq.SUBSCRIBE, t.encode('utf-8'))

    print("Sent message", topics)
    message_num += 1
    msg = Message(topics, query.atomic_functions[1],0, time.time(), message_num, query_num)


    sock_to_sensors.send_multipart([b'topics_all', pickle.dumps(msg)])
    time.sleep(1)
    sock_to_sensors.send_multipart([b'topics_all', pickle.dumps(msg)])



    #try:
    while True:
        topic, msg = sock_from_sensors.recv_multipart()
        print("received", topic.decode('utf-8'))
        msg = pickle.loads(msg)
        print(msg)
        #pdb.set_trace()
        #visitor.update_signal(msg[0].time, msg)
        #output = visitor.visit(tree)
        #if output:
        #    print("Query satisfied at T=" + str(msg.time))
    #except:
    #    print("Error thread.")


if __name__ == "__main__":

    #HOST, PORT = "localhost", 5005
    #listener = Listener(query_address, authkey=b'secret password')
    #server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)

    UNIX_HOST = "/tmp/query_server"
    try:
        os.unlink(UNIX_HOST)
    except:
        print("no file")
    server_query = ThreadedUnixServer(UNIX_HOST, ThreadedUnixRequestHandler)
    #with server:
    #ip, port = server.server_address

    """
    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=server.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    """

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread_query = threading.Thread(target=server_query.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread_query.daemon = True
    server_thread_query.start()

    #print("Server loop running in thread:", server_thread.name)

    #client(ip, port, "Hello World 1")
    #client(ip, port, "Hello World 2")
    #client(ip, port, "Hello World 3")

    query_path = "current_query.txt"
    with open(query_path, "r") as f:
        data = f.read()

    atomic_functions = ['watchbox', [[400, 0, 800, 600]]]#[data]
    #visitor, tree = get_query(data)
    #atomic_functions = get_ordered_atomic_functions(visitor, tree)
    print(atomic_functions)
    query = Query(atomic_functions, 0)
    queries.append(query)
    
    visitor=''
    tree=''
    x = threading.Thread(target=setup_thread, args=(query,visitor,tree,num_queries,), daemon=True)
    num_queries += 1
    x.start()

    while True:
        time.sleep(1)





