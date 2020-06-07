from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import scipy.ndimage.filters
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import scipy.ndimage.filters
import time
import datetime
import os.path
import io

import os

os.environ["PATH"] += os.pathsep + "."


# % matplotlibinline


def get_kernel(kernel_size, blur=1 / 20, halo=.001):
    """
    Create an (n*2+1)x(n*2+1) numpy array.
    Output can be used as the kernel for convolution.
    """

    # generate x and y grids
    x, y = np.mgrid[0:kernel_size * 2 + 1, 0:kernel_size * 2 + 1]

    center = kernel_size + 1  # center pixel
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)  # distance from center

    # now compute the kernel. This function is a bit arbitrary.
    # adjust this to get the effect you want.
    kernel = np.exp(-r / kernel_size / blur) + (1 - r / r[center, 0]).clip(0) * halo
    return kernel


def add_lines(image_array, xys, width=1, weights=None):
    """
    Add a set of lines (xys) to an existing image_array
    width: width of lines
    weights: [], optional list of multipliers for lines.
    """

    for i, xy in enumerate(xys):  # loop over lines
        # create a new gray scale image
        image = Image.new("L", (image_array.shape[1], image_array.shape[0]))

        # draw the line
        ImageDraw.Draw(image).line(xy, 200, width=width)

        # convert to array
        new_image_array = np.asarray(image, dtype=np.uint8).astype(float)

        # apply weights if provided
        if weights is not None:
            new_image_array *= weights[i]

        # add to existing array
        image_array += new_image_array

    # convolve image
    new_image_array = scipy.ndimage.filters.convolve(image_array, get_kernel(width * 4))
    return new_image_array


def to_image(array, hue=.62):
    """converts an array of floats to an array of RGB values using a colormap"""

    # apply saturation function
    image_data = np.log(array + 1)

    # create colormap, change these values to adjust to look of your plot
    saturation_values = [[0, 0], [.75, .68], [.78, .87], [0, 1]]
    colors = [hsv_to_rgb([hue, x, y]) for x, y in saturation_values]
    cmap = LinearSegmentedColormap.from_list("my_colormap", colors)

    # apply colormap
    out = cmap(image_data / image_data.max())

    # convert to 8-bit unsigned integer
    out = (out * 255).astype(np.uint8)
    return out


def get_min_max(bike_data):
    min_lat = bike_data["Start Station Latitude"].min()
    max_lat = bike_data["Start Station Latitude"].max()
    max_lon = bike_data["Start Station Longitude"].max()
    min_lon = bike_data["Start Station Longitude"].min()
    return min_lat, max_lat, min_lon, max_lon


def latlon_to_pixel(lat, lon, image_shape, bounds):
    min_lat, max_lat, min_lon, max_lon = bounds

    # longitude to pixel conversion (fit data to image)
    delta_x = image_shape[1] / (max_lon - min_lon)

    # latitude to pixel conversion (maintain aspect ratio)
    delta_y = delta_x / np.cos(lat / 360 * np.pi * 2)
    pixel_y = (max_lat - lat) * delta_y
    pixel_x = (lon - min_lon) * delta_x
    return (pixel_y, pixel_x)


def row_to_pixel(row, image_shape, columns=None):
    """
    convert a row (1 trip) to pixel coordinates
    of start and end point
    """
    start_y, start_x = latlon_to_pixel(row["Start Station Latitude"],
                                       row["Start Station Longitude"], image_shape)
    end_y, end_x = latlon_to_pixel(row["End Station Latitude"],
                                   row["End Station Longitude"], image_shape)
    xy = (start_x, start_y, end_x, end_y)
    return xy


def plot_station_counts(trip_counts, zoom_start=13):
    # generate a new map
    folium_map = folium.Map(location=[40.738, -73.98],
                            zoom_start=zoom_start,
                            tiles="CartoDB dark_matter",
                            width="100%")

    # for each row in the data, add a cicle marker
    for index, row in trip_counts.iterrows():
        # calculate net departures
        net_departures = (row["Departure Count"] - row["Arrival Count"])

        # generate the popup message that is shown on click.
        popup_text = "{}<br> total departures: {}<br> total arrivals: {}<br> net departures: {}"
        popup_text = popup_text.format(row["Start Station Name"],
                                       row["Arrival Count"],
                                       row["Departure Count"],
                                       net_departures)

        # radius of circles
        radius = net_departures / 20

        # choose the color of the marker
        if net_departures > 0:
            # color="#FFCE00" # orange
            # color="#007849" # green
            color = "#E37222"  # tangerine
        else:
            # color="#0375B4" # blue
            # color="#FFCE00" # yellow
            color = "#0A8A9F"  # teal

        # add marker to the map
        folium.CircleMarker(location=(row["Start Station Latitude"],
                                      row["Start Station Longitude"]),
                            radius=radius,
                            color=color,
                            popup=popup_text,
                            fill=True).add_to(folium_map)
    return folium_map


def get_locations(bike_data):
    locations = bike_data.groupby("Start Station ID").first()
    locations = locations.loc[:, ["Start Station Latitude",
                                  "Start Station Longitude",
                                  "Start Station Name"]]
    return locations


def get_trip_counts_by_hour(selected_hour, bike_data):
    # make a DataFrame with locations for each bike station

    locations = get_locations(bike_data)
    # select one time of day
    subset = bike_data[bike_data["hour"] == selected_hour]

    # count trips for each destination
    departure_counts = subset.groupby("Start Station ID").count()
    departure_counts = departure_counts.iloc[:, [0]]
    departure_counts.columns = ["Departure Count"]

    # count trips for each origin
    arrival_counts = subset.groupby("End Station ID").count().iloc[:, [0]]
    arrival_counts.columns = ["Arrival Count"]

    # join departure counts, arrival counts, and locations
    trip_counts = departure_counts.join(locations).join(arrival_counts)
    return trip_counts


def add_alpha(image_data):
    """
    Uses the Value in HSV as an alpha channel.
    This creates an image that blends nicely with a black background.
    """

    # get hsv image
    hsv = rgb_to_hsv(image_data[:, :, :3].astype(float) / 255)

    # create new image and set alpha channel
    new_image_data = np.zeros(image_data.shape)
    new_image_data[:, :, 3] = hsv[:, :, 2]

    # set value of hsv image to either 0 or 1.
    hsv[:, :, 2] = np.where(hsv[:, :, 2] > 0, 1, 0)

    # combine alpha and new rgb
    new_image_data[:, :, :3] = hsv_to_rgb(hsv)
    return new_image_data


def create_image_map(image_data, bounds):
    min_lat, max_lat, min_lon, max_lon = bounds
    folium_map = folium.Map(location=[40.738, -73.98],
                            zoom_start=13,
                            tiles="CartoDB dark_matter",
                            width='100%')

    # create the overlay
    map_overlay = add_alpha(to_image(image_data))

    # compute extent of image in lat/lon
    aspect_ratio = map_overlay.shape[1] / map_overlay.shape[0]
    delta_lat = (max_lon - min_lon) / aspect_ratio * np.cos(min_lat / 360 * 2 * np.pi)

    # add the image to the map
    img = folium.raster_layers.ImageOverlay(map_overlay,
                                            bounds=[(max_lat - delta_lat, min_lon), (max_lat, max_lon)],
                                            opacity=1,
                                            name="Paths")

    img.add_to(folium_map)
    folium.LayerControl().add_to(folium_map)

    # return the map
    return folium_map


def get_path_progress(trips, image_time):
    """ return a series of numbers between 0 and 1
    indicating the progress of each trip at the given time"""

    trip_duration = trips["Stop Time"] - trips["Start Time"]
    path_progress = (image_time - trips["Start Time"]).dt.total_seconds() / trip_duration.dt.total_seconds()
    return path_progress


def get_current_position(trips, progress):
    """ Return Latitude and Longitude for the 'current position' of each trip.
    Paths are assumed to be straight lines between start and end.
    """

    current_latitude = trips["Start Station Latitude"] * (1 - progress) + \
                       trips["End Station Latitude"] * progress
    current_longitude = trips["Start Station Longitude"] * (1 - progress) + \
                        trips["End Station Longitude"] * progress
    return current_latitude, current_longitude


def get_active_trips(image_time, bike_data, image_shape, line_len=.1):
    """ Return pixel coordinates only for trips that have started and
    not yet completed for the given time.
    """

    bounds = get_min_max(bike_data)

    active_trips = bike_data[(bike_data["Start Time"] <= image_time)]
    active_trips = active_trips[(active_trips["Stop Time"] >= image_time)].copy()

    progress = get_path_progress(active_trips, image_time)

    current_latitude, current_longitude = get_current_position(active_trips, progress)
    start_latitude, start_longitude = get_current_position(active_trips, np.clip(progress - line_len, 0, 1))

    start_y, start_x = latlon_to_pixel(start_latitude,
                                       start_longitude,
                                       image_shape,
                                       bounds)

    end_y, end_x = latlon_to_pixel(current_latitude,
                                   current_longitude,
                                   image_shape,
                                   bounds)
    xys = list(zip(start_x, start_y, end_x, end_y))
    weights = np.clip((1 - progress.values) * 100, 0, 1)

    return xys, weights


def row_to_pixel(row, image_shape, bounds):
    """
    convert a row (1 trip) to pixel coordinates
    of start and end point
    """
    start_y, start_x = latlon_to_pixel(row["Start Latitude"],
                                       row["Start Longitude"],
                                       image_shape,
                                       bounds)
    end_y, end_x = latlon_to_pixel(row["End Latitude"],
                                   row["End Longitude"],
                                   image_shape,
                                   bounds)
    xy = (start_x, start_y, end_x, end_y)
    return xy


def get_image_map(frame_time, bike_data):
    """Create the folium map for the given time"""

    image_data = np.zeros((900 * 2, 400 * 2))
    bounds = get_min_max(bike_data)

    # plot the current locations
    xys, weights = get_active_trips(frame_time, bike_data, image_data.shape, line_len=.01)
    image_data = add_lines(image_data, xys, weights=weights * 20, width=4)

    #  plot the paths
    xys, weights = get_active_trips(frame_time, bike_data, image_data.shape, line_len=1)
    image_data = add_lines(image_data, xys, weights=weights * 10, width=2)

    # generate and return the folium map.
    return create_image_map(image_data, bounds)


def create_image_map(image_data, bounds):
    min_lat, max_lat, min_lon, max_lon = bounds
    folium_map = folium.Map(location=[40.738, -73.98],
                            zoom_start=13,
                            tiles="CartoDB dark_matter",
                            width='100%')

    # create the overlay
    map_overlay = add_alpha(to_image(image_data))

    # compute extent of image in lat/lon
    aspect_ratio = map_overlay.shape[1] / map_overlay.shape[0]
    delta_lat = (max_lon - min_lon) / aspect_ratio * np.cos(min_lat / 360 * 2 * np.pi)

    # add the image to the map
    img = folium.raster_layers.ImageOverlay(map_overlay,
                                            bounds=[(max_lat - delta_lat, min_lon), (max_lat, max_lon)],
                                            opacity=1,
                                            name="Paths")

    img.add_to(folium_map)
    folium.LayerControl().add_to(folium_map)

    # return the map
    return folium_map


def go_paths_frame(params, bike_data):
    """Similar to go_arrivals_frame.
    Generate the image, add annotations, and save image file."""
    i, frame_time = params

    my_frame = get_image_map(frame_time, bike_data)
    png = my_frame._to_png()

    image = Image.open(io.BytesIO(png))
    draw = ImageDraw.ImageDraw(image)
    font = ImageFont.truetype("Roboto-Light.ttf", 30)

    # add date and time of day text
    draw.text((20, image.height - 50),
              "time: {}".format(frame_time),
              fill=(255, 255, 255),
              font=font)

    # draw title
    draw.text((image.width - 450, 20),
              "Paths of Individual Bike Trips",
              fill=(255, 255, 255),
              font=font)

    # write to a png file
    dir_name = "path_frames"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    image.save(os.path.join(dir_name, "frame_{:0>5}.png".format(i)), "PNG")
    return image

def main():
    print("read data")
    bike_data = pd.read_csv("201610-citibike-tripdata.csv")
    print("Data readed")
    print("Start to_datetime")
    bike_data["Start Time"] = pd.to_datetime(bike_data["Start Time"])
    bike_data["Stop Time"] = pd.to_datetime(bike_data["Stop Time"])
    # bike_data["hour"] = bike_data["Start Time"].map(lambda x: x.hour)
    bike_data["hour"] = bike_data["Start Time"].dt.hour
    print("start claclulating 1. image")
    get_image_map(pd.to_datetime('2016-10-05 09:00:00'), bike_data)
    print("start claclulating 2. image")
    go_paths_frame((1, pd.to_datetime('2016-10-05 09:00:00')), bike_data)
