#! /bin/bash

cd ../
source venv/bin/activate
cd simulator/
scenic pedestrian.scenic --simulate --param num_scenario 0 --param num_extra_pedestrians 0 --param output_dir camera_img --param bind_address $1 --param cameras_on "1,2"
