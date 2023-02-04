class Data():
    def __init__(self, data, topic):
        #self.time = time
        #self.location = location
        self.data = data
        self.topic = topic



class Location():
    def __init__(self, view):
        #self.sensor = sensor
        self.view = view

class Message():
    def __init__(self,topics,arguments,area,time,message_number,query_number):
        self.time = time
        self.topics = topics
        self.area = area
        self.message_number = message_number
        self.query_number = query_number
        self.arguments = arguments
