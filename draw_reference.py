import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm

try:
    sys.path.append('CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg')
    sys.path.append('CARLA_0.9.10/PythonAPI/carla/')
        #'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


# Search for a particular description in our points, and get it as a dictionary result
def find_data_for_desc(points_data, desc):
    for x in points_data:
        if desc in x:
            return get_data_point_as_dict(x)
    return -1

# Get the current data point as a dictionary result
def get_data_point_as_dict(point):

    # Create a dictionary result
    keys = ["x", "y", "z", "pitch", "yaw", "roll", "desc"]
    result = {}
    for i in range(0, len(keys)):
        result[keys[i]] = point[i]

    return result


# Read in all our lines and get the data structure
def read_data(points_filepath="carla_code/points.txt"):

    points_data = []

    # Open the points file where we store all our waypoints/camera positions/etc
    with open(points_filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            row = [x.strip() for x in row]
            if not row:
                continue
            # We ignore rows with '#' comments
            comments = [1 if "#" in x else 0 for x in row]
            if comments[0] == 1:  # The whole row is a comment, and we ignore
                continue
            elif 1 in comments: # There is a comment at a later point
                comment_index = comments.index(1)
                last_obj = row[comment_index].split("#")[0]
                row = row[:comment_index]
                row.append(last_obj.strip())

            # Do some basic error checking
            assert len(row) == 7

            #  Appent our data points
            desc = row[-1]
            row = [float(x) for x in row[:-1]]
            points_data.append(row + [desc])

    return points_data


# Takes in a point, and scales/transforms it into the image space
def scale_and_translate(point, origin, scale_x, scale_y):

    # First, we subtract from the origin in the point domain to transform
    new_point_x = point["x"] - origin["x"]
    new_point_y = origin["y"] - point["y"]

    # Scale to the image domain
    new_point_x = new_point_x*scale_x
    new_point_y = new_point_y*scale_y

    return new_point_x, new_point_y




# Draw over our reference image
#  Note that the coordinates of an image (0,0) start from the top left (not bottom left)
def draw_over_reference(points_data, save_filename, colors, reference_image_filepath="carla_code/reference.png", \
    save_fig_dir = "carla_code/overlays"):

    # First, open our image
    img = plt.imread(reference_image_filepath)
    implot = plt.imshow(np.flipud(img), origin='lower')
    image_max_y, image_max_x  = implot.get_size()


    # Now, figure out our corners based on the b1,b2 descriptors
    points_b1 = find_data_for_desc(points_data, "b1") # Taken from bottom left of image
    points_b2 = find_data_for_desc(points_data, "b2") # Taken from top right of image

    origin = points_b1 #  This will be our origin (bottom left)

    # Now normalize it w.r.t. the image
    points_width = points_b2["x"] - points_b1["x"]
    points_height = points_b1["y"] - points_b2["y"]

    # Find the scaling factor for x and y to move from the points space to image space
    scale_x = image_max_x / points_width
    scale_y = image_max_y / points_height

    # Now we go through all of our points and plot them
    x,y,labels = [], [], []
    for i,point in enumerate(points_data):

        # Plot data
        curr_x, curr_y = scale_and_translate(get_data_point_as_dict(point), origin, scale_x, scale_y)
        curr_label = point[-1]

        x.append(curr_x)
        y.append(curr_y)
        labels.append(curr_label)

        if not colors:
            # Plot our points
            if "tc" in curr_label:
                plt.scatter(x=curr_x, y=curr_y, c='b')
                plt.annotate(curr_label, (curr_x, curr_y), fontsize=7)
            elif "r" in curr_label:
                plt.scatter(x=curr_x, y=curr_y, c='g', s=2)
                plt.annotate(curr_label, (curr_x, curr_y), fontsize=5)
            else:
                plt.scatter(x=curr_x, y=curr_y, c='r')
                plt.annotate(curr_label, (curr_x, curr_y), fontsize=7)
        else:
            plt.scatter(x=curr_x, y=curr_y, c=colors[i], s=2)
            plt.annotate(curr_label, (curr_x, curr_y), fontsize=5)



    if not os.path.exists(save_fig_dir):
        os.mkdir(save_fig_dir)
    # Also save our image to file
    plt.savefig(save_fig_dir + '/' + save_filename + '.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create a mapping of rows and dictionaries
def create_waypoint_road_dict(waypoints):

    # Now we take our waypoints generated from the map, and get the corresponding roads.
    # road_dict is {road_id: [waypoint, waypoint, etc.]}
    road_dict = {}
    for wp in waypoints:
        if wp.road_id in road_dict:
            road_dict[wp.road_id].append(wp)
        else: # Not in road dict, so create the list
            road_dict[wp.road_id] = [wp]

    return road_dict

# Create many images that describe the different waypoints based on roads
def overlay_waypoints(road_dict, points_data):

    # For every road, we select all of its waypoints
    for rid in tqdm(road_dict.keys()):
        # chosen_waypoint = np.random.choice(road_dict[rid])

        roadname = "r"+str(rid)

        road_waypoints = []
        colors = ["b" for x in points_data]
        for wp in road_dict[rid]:
            # We save this to our existing points
            location = wp.transform.location
            new_data = [location.x, location.y, location.z, -1, -1,-1, roadname]
            # waypoint = get_data_point_as_dict(new_data)
            road_waypoints.append(new_data)
            # Also be sure to get the lane_id, green for positive, red for negative
            if wp.lane_id > 0:
                colors.append("g")
            else:
                colors.append("r")


        # We draw this as a new overlay
        draw_over_reference(points_data + road_waypoints, roadname, colors)


if __name__ == '__main__':

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Once we have a client we can retrieve the world that is currently
    # running.
    world = client.get_world()
    # Check the current loaded world - if it is not Town10, we load it.
    if world.get_map().name != "Town10HD":
        world = client.load_world('Town10HD')

    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    fixed_delta_seconds = 0.05 # 20 FPS for 0.05
    settings.fixed_delta_seconds = fixed_delta_seconds
    world.apply_settings(settings)

    # Generate the map waypoints
    waypoints = client.get_world().get_map().generate_waypoints(distance=2.0)
    # This is meant for mapping waypoints to roads (it's how we decide our paths)
    road_dict = create_waypoint_road_dict(waypoints)

    # If we actually want to create image files of our waypoints plotted on the map
    points_data = read_data("carla_code/points.txt")
    overlay_waypoints(road_dict, points_data)
