# DistributedCE
Distributed Complex Event Detection Architecture

## Setup

1. Run `git clone --recurse-submodules https://github.com/nesl/DistributedCE.git`
2. Change directory: `cd simulator/`
3. Install CARLA 0.9.10 with the Additional Maps. Extract the CARLA files and put them under a new directory with the name of **CARLA_0.9.10**.
4. Install [Poetry](https://python-poetry.org/) into your system
5. Install Scenic by getting into the Scenic folder and using `poetry install -E dev`
6. Clone [CARLA-2DBBox](https://github.com/MukhlasAdib/CARLA-2DBBox) and name it bbox_annotation.
7. Change directory: `cd ../network`
8. Clone [Mininet](https://github.com/mininet/mininet) and move the **mininet/mininet** subfolder to the current directory.
9. Run `sudo python3 test_mininet.py`
10. When the interactive console appears, run `source mininet_commands`
11. In each respective open xterm, run the respective *h?.sh* script, starting with *h1_1.sh* and *h1_2.sh*. For *h1_2.sh*, *h2.sh* and *h4.sh* wait until the output says *Setting up Server...* to continue executing the other *h?.sh* scripts.
12. You can also try running the *h?.sh* script without mininet in different terminal tabs but keep in mind you need to change the addresses.

Note: There are two *requirements.txt* files for two different virtual environments, one in the main directory for *venv* and the other in *detection* directory for *yolo*.

## Changes to Scenic

* Added the capability to specify both depth and RGB cameras, as shown in the code snippet below. Because we want the bounding boxes of any objects we are interesed in, we create an RGB Camera coupled with a depth camera to do this, as needed by the functions used by the **CARLA-2DBBox** library. We specify the locations of these cameras, with *c* being a vector of the form `[x,y,z,pitch,yaw,roll,camera id]`, and also define a behavior that takes the path where images and the bounding box information are going to be saved, and will make use of the *actors_bb* list with all the objects one wants to track.

```
behavior CameraBehavior(path):
    while True:
        take GetBoundingBox(actors_bb,path)


depth_camera = depthCamera at c[0] @ -c[1],
    with elevation c[2],
    with pitch c[3],
    with yaw c[4],
    with roll 0

rgbCamera at depth_camera,
    with elevation c[2],
    with pitch c[3],
    with yaw c[4],
    with roll 0,
    with depth depth_camera, 
    with camera_id int(c[-1][-1]),
    with behavior CameraBehavior(output_dir)


```


* The *SetWalkDestination* action is implemented in order to control where do pedestrians go. You can use it as follows: `take SetWalkDestination(destination)`. Check the *WalkAround* behavior for more information on how to use it.

* And Object_Dummy auxiliary class was created in order to instantiate or eliminate any object of type *static.prop* (all objects that are not pedestrians or cars) at any point of the simulation. A code snippet to instantiate a box is shown below, an *Object_Dummy* object is initialized by specifying its blueprint, position, heading, and elevation.

```
obj_dumm = Object_Dummy(Uniform(*blueprints.boxModels),box_destination,0,0)
simulation().createObjectInSimulator(obj_dumm)
```

&nbsp;To destroy the object:

`obj_dumm.carlaActor.destroy()`

* All *static.prop* objects have been associated with a static bounding box in order to work with the **CARLA-2DBBox** library.

* Control over vehicle's trajectories can be obtained using the *BasicAgent* class that provides CARLA. An example is provided with the *CarFoBehavior* behavior.

* To run any scenario use run_scenario(num_scenario=3,num_extra_pedestrians=100, output_dir="camera_img/")
