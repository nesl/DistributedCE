#! /bin/bash

cd ../
su nesl -c simulator/CARLA_0.9.10/CarlaUE4.sh &
source venv/bin/activate
cd simulator/
scenic pedestrian.scenic --simulate
