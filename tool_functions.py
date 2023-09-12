import re
import os
import numpy as np
import tifffile as tiflib
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from skimage.draw import polygon
from tqdm import tqdm


# list all filenames in a directory including the provided path
def getListOfFiles(dirName, pattern='', recursive=True, addDirs=False):
    # create a list of file and sub-directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if not ('.DS' in fullPath):
            # print(fullPath)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                if recursive:
                    allFiles = allFiles + getListOfFiles(fullPath, pattern, recursive, addDirs)
                if addDirs:
                    allFiles.append(fullPath)
            else:
                if not (re.search(pattern, fullPath.lower()) == None):
                    allFiles.append(fullPath)

    return np.sort(allFiles)


def loadTIF(filename, Frame=None, Verbose=True, Only_Info=False):
    I = []
    if not Only_Info:
        if Frame == None:
            I = tiflib.imread(filename).swapaxes(0, 2)
        else:
            I = tiflib.imread(filename, key=Frame)

    with tiflib.TiffFile(filename) as tif:
        imagej_metadata = tif.imagej_metadata

    if Verbose:
        print(imagej_metadata)
    if imagej_metadata is None:
        width = 640
        height = 480
        frames = 1
    else:
        # is this true for all sources?
        width = imagej_metadata['Labels'][0].split('\n')[1].split(" ")[2][1:-1]
        height = imagej_metadata['Labels'][0].split('\n')[1].split(" ")[3][:-2]
        frames = imagej_metadata['slices']
    return I, {'image_type': 'uint16', 'width': int(width), 'height': int(height), 'frames': frames}


# interactive masking
class PolygonDrawer(object):
    def __init__(self, img):
        self.window_name = "Crop image"  # Name for our window
        self.clone = img.copy()
        self.done = False  # Flag signalling we're done
        self.current = (0, 0)  # Current position, so we can draw the line-in-progress
        self.points = []  # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    @property
    def run(self):
        FINAL_LINE_COLOR = (255, 255, 255)
        WORKING_LINE_COLOR = (127, 127, 127)
        self.points.clear()
        # Let's create our working window and set a mouse callback to handle events
        cv.namedWindow(self.window_name)
        cv.imshow(self.window_name, self.clone)
        print('hello')
        cv.waitKey(1)
        cv.setMouseCallback(self.window_name, self.on_mouse)
        canvas = self.clone.copy()
        while (not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                # And  also show what the current segment would look like
                cv.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv.waitKey(50) == 27:  # ESC hit
                self.done = True
        # User finised entering the polygon points, so let's make the final drawing
        canvas = self.clone.copy()
        # of a filled polygon
        if (len(self.points) > 0):
            stencil = np.zeros(canvas.shape).astype(canvas.dtype)
            cv.fillPoly(stencil, np.array([self.points]), [255, 255, 255])
            canvas = cv.bitwise_and(canvas, stencil)
        # And show it
        cv.imshow(self.window_name, canvas)
        # Waiting for the user to accept or reject
        print("Press any key to accept the crop, press d to start over")
        if cv.waitKey(0) == 100:  # d hit
            self.done = False
            return -1
        else:
            cv.destroyWindow(self.window_name)
            return self.points


class PolygonDrawerMatplotlib(object):
    def __init__(self, img):
        self.original_img = img.copy()
        self.setup()

    def setup(self):
        self.img = self.original_img.copy()
        self.mask = None
        self.points = []  # List of points defining our polygon
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img)
        self.line, = self.ax.plot([], [], lw=2)  # Initialize an empty line
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Add a button for accepting the result
        self.accept_button = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.button = plt.Button(self.accept_button, 'Accept')
        self.button.on_clicked(self.accept_result)

        # Add a button for rejecting the result
        self.reject_button = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.reject_btn = plt.Button(self.reject_button, 'Reject')
        self.reject_btn.on_clicked(self.reject_result)

        self.result = None

    def on_click(self, event):
        ix, iy = event.xdata, event.ydata
        if ix is None or iy is None or ix < 0.5 or iy < 0.5:
            return
        if event.button == 1:  # Left click
            print(f'Adding point with position ({ix}, {iy})')
            self.points.append((ix, iy))
            self.line.set_data(list(zip(*self.points)))
            self.fig.canvas.draw()

        elif event.button == 3:  # Right click
            print(f'Completing polygon with {len(self.points)} points.')
            self.points.append(self.points[0])  # Close the polygon
            self.line.set_data(list(zip(*self.points)))
            self.mask_image()
            self.ax.imshow(self.img)
            self.fig.canvas.draw()

    def mask_image(self):
        # Create a blank black image with the same dimensions
        # this option works for i channel images, if more channels then use self.original_img[:,:,0]
        self.mask = np.zeros(self.original_img.shape)
        print('mask shape', self.mask.shape)
        # Get the coordinates of the polygon as row and column indices
        row_coords, col_coords = zip(*self.points)
        # Create a mask for the polygon area
        rr, cc = polygon(col_coords, row_coords, self.original_img.shape)
        self.mask[rr, cc] = 1
        # Mask the original image
        print('img shape', self.img.shape)
        self.img = self.original_img * np.squeeze(self.mask)

    def accept_result(self, event):
        self.result = "accepted"
        plt.close()

    def reject_result(self, event):
        self.result = "rejected"
        plt.close()

    def run(self):
        while True:
            plt.show()
            if self.result == "accepted":
                return self.mask
            elif self.result == "rejected":
                self.setup()


# standard moving average filter
def movAverage(values, filter_halfwidth):
    res = []
    for i in range(0, values.shape[0]):
        res.append(np.average(values[max(0, i - filter_halfwidth):min(values.shape[0], i + filter_halfwidth)]))
    return np.asarray(res)


# 2 pass moving average filter
# 1st pass standard
# 2nd pass performs an moving average using the inverse distance to the 1st pass as weights
# the distance can be amplified using fac
def adaptiveMovAverage(values, filter_halfwidth, fac=1):
    filtered_values = movAverage(values, filter_halfwidth)
    res = []

    for i in range(0, values.shape[0]):
        subvalues = values[max(0, i - filter_halfwidth):min(values.shape[0], i + filter_halfwidth)]
        filtered_subvalues = filtered_values[max(0, i - filter_halfwidth):min(values.shape[0], i + filter_halfwidth)]
        weight = 1 / (fac * np.abs(subvalues - filtered_subvalues) + 1)

        res.append(np.sum(np.multiply(weight, subvalues)) / np.sum(weight))

    return np.asarray(res)


def find_intervals(values, peaks, threshold):
    intervals = []
    for i in range(1, peaks.shape[0]):
        v = values[peaks[i]]
        pos = peaks[i]
        # print('v=',v,' pos=',pos)
        abb = False
        for j in range(peaks[i], peaks[i - 1], -1):
            # print('j=',j)
            if values[j] < v and not (abb):
                v = values[j]
                pos = j
            elif values[j] < threshold:
                abb = True
        intervals.append(int(pos))
    return np.array(intervals)
