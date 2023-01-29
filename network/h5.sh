#! /bin/bash

cd ../
source venv/bin/activate
cd detection/
python forwarder.py --address_source 10.0.0.1 --address_sink 10.0.0.4 --port_source 55002 --port_sink 58002
