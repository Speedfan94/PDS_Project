import pandas as pd
import numpy as np
import folium
from PIL import Image, ImageDraw, ImageFont
from folium import plugins
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, hsv_to_rgb
import scipy.ndimage.filters
import io
import os
from .. import io


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
    min_lat = bike_data["Latitude_start"].min()
    max_lat = bike_data["Latitude_start"].max()
    max_lon = bike_data["Longitude_start"].max()
    min_lon = bike_data["Longitude_start"].min()
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


def row_to_pixel(row,image_shape, columns=None):
    """
    convert a row (1 trip) to pixel coordinates
    of start and end point
    """
    start_y, start_x = latlon_to_pixel(row["Latitude_start"],
                                       row["Longitude_start"], image_shape)
    end_y, end_x = latlon_to_pixel(row["Latitude_end"],
                                   row["Longitude_end"], image_shape)
    xy = (start_x, start_y, end_x, end_y)
    return xy



def start_interactive_maps():
    bike_data = pd.read_csv("D:\Eigene Dateien\OneDrive\Dokumente\GitHub\PDS_Project/nextbike/data/output/Trips.csv")
    bike_data["Start_Time"] = pd.to_datetime(bike_data["Start_Time"])
    bike_data["End_Time"] = pd.to_datetime(bike_data["End_Time"])
    # TODO: to be removed, if work
    # bike_data["hour"] = bike_data["Start_Time"].map(lambda x: x.hour)
    return bike_data


def plot_station_counts(trip_counts, zoom_start=13):
    # generate a new map
    folium_map = folium.Map(location=[49.452633, 11.095934],
                            zoom_start=zoom_start,
                            tiles="CartoDB dark_matter",
                            width="100%")

    # for each row in the data, add a circle marker
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
        radius = net_departures / 10

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
        folium.CircleMarker(location=(row["Latitude_start"],
                                      row["Longitude_start"]),
                            radius=radius,
                            color=color,
                            popup=popup_text,
                            fill=True).add_to(folium_map)
    return folium_map


def get_locations(bike_data):
    locations = bike_data.groupby("p_uid_start").first()
    locations = locations.loc[:, ["Latitude_start",
                                  "Longitude_start"]]
                                  # "p_uid_start"]]
    return locations


def get_trip_counts_by_hour(selected_hour, bike_data):
    # make a DataFrame with locations for each bike station
    locations = get_locations(bike_data)

    # select one time of day
    subset = bike_data[bike_data["hour"] == selected_hour]

    # count trips for each destination
    departure_counts = subset.groupby("p_uid_start").count()
    departure_counts = departure_counts.iloc[:, [0]]
    departure_counts.columns = ["Departure Count"]

    # count trips for each origin
    arrival_counts = subset.groupby("p_uid_end").count().iloc[:, [0]]
    arrival_counts.columns = ["Arrival Count"]

    # join departure counts, arrival counts, and locations
    trip_counts = departure_counts.join(locations).join(arrival_counts)
    return trip_counts

# print a sample to check our code works
# get_trip_counts_by_hour().head()


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
    aspect_ratio = map_overlay.shape[1]/map_overlay.shape[0]
    delta_lat = (max_lon-min_lon)/aspect_ratio*np.cos(min_lat/360*2*np.pi)

    # add the image to the map
    img = folium.raster_layers.ImageOverlay(map_overlay,
                                            bounds=[(max_lat-delta_lat, min_lon), (max_lat, max_lon)],
                                            opacity=1,
                                            name="Paths")

    img.add_to(folium_map)
    folium.LayerControl().add_to(folium_map)

    # return the map
    return folium_map


def interpolate(df1, df2, x):
    """return a weighted average of two dataframes"""
    df = df1 * (1 - x) + df2 * x
    return df.replace(np.nan, 0)


def get_trip_counts_by_minute(float_hour, data):
    """get an interpolated dataframe for any time, based
    on hourly data"""

    columns = ["Latitude_start",
               "Longitude_start",
               "Departure Count",
               "Arrival Count"]
    df1 = get_trip_counts_by_hour(int(float_hour), data)
    df2 = get_trip_counts_by_hour(int(float_hour) + 1, data)

    df = interpolate(df1.loc[:, columns],
                     df2.loc[:, columns],
                     float_hour % 1)

    df["p_uid_start"] = df1["p_uid_start"]
    return df

#TODO: SAvePath rechecking
def go_arrivals_frame(i, hour_of_day, save_path):
    # create the map object
    data = get_trip_counts_by_minute(hour_of_day, bike_data)
    my_frame = plot_station_counts(data, zoom_start=14)

    # generate the png file as a byte array
    png = my_frame._to_png()

    #  now add a caption to the image to indicate the time-of-day.
    hour = int(hour_of_day)
    minutes = int((hour_of_day % 1) * 60)

    # create a PIL image object
    image = Image.open(io.BytesIO(png))
    draw = ImageDraw.ImageDraw(image)

    # load a font
    font = ImageFont.truetype("Roboto-Light.ttf", 30)

    # draw time of day text
    draw.text((20, image.height - 50),
              "time: {:0>2}:{:0>2}h".format(hour, minutes),
              fill=(255, 255, 255),
              font=font)

    # draw title
    draw.text((image.width - 400, 20),
              "Net Arrivals vs Time of Day",
              fill=(255, 255, 255),
              font=font)

    # write to a png file
    # io.output.save_fig(image, "frame_{:0>5}.png")
    filename = os.path.join(save_path, "frame_{:0>5}.png".format(i))
    image.save(filename, "PNG")
    return image


def main_interactive_maps():
    bike_data = start_interactive_maps()
    plot_station_counts(get_trip_counts_by_minute(9.5, bike_data), zoom_start=14)

    dir_name = 'frames'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    go_arrivals_frame(1, 8.5)

    arrival_times = np.arange(6, 23, .2)
    for i, hour in enumerate(arrival_times):
        go_arrivals_frame(i, hour, "frames")


