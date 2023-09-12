import tool_functions as tf
import paths_and_flags as pf
import numpy as np
from skimage.measure import label, regionprops
from skimage import measure
from skimage import img_as_float, img_as_ubyte
from skimage import morphology
from scipy.ndimage import distance_transform_edt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import use as mpl_use
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch
import cv2
mpl_use('MacOSX')
###########################################################
# pattern specific parameters
number_of_calib_pattern = 20
CHECKERBOARD = (8, 6)
w = 640
h = 480
number_of_calib_pattern = len(pf.import_calib_folder)

camera_pairs = [[1, 0], [1, 2], [1, 3]]

###########################################################
# main calibration function

def do_calibration(Verbose = False):

    if (Verbose):
        pattern_id_to_display = 0
        filenames = tf.getListOfFiles(str(pf.import_calib_folder[pattern_id_to_display]), 'tif', False, False)
        cameras = []
        fig = plt.figure(figsize=(10, 7))
        i = 1
        for file in filenames:
            Img, info = tf.loadTIF(file, 0, False, False)
            ax = plt.subplot(2, 2, i);
            ax.imshow(Img, cmap='Greys_r')
            ax.axis('off')
            ax.set_title(file.split('/')[-1])
            i = i + 1
        plt.show()

    print(pf.import_calib_folder)
    print(number_of_calib_pattern, ' calibration pattern folder found')

    rotation_invariant = True
    camera_size_factor = [0, 1, 0, 1]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,
                                                                                      2) * 2.06  # physical checkerboard 5mm resolution
    prev_img_shape = None
    pattern = []
    for i in tf.tqdm(range(1, len(pf.import_calib_folder) + 1), desc='reading pattern'):
        filenames = tf.getListOfFiles(str(calib_folder_names[i - 1]), 'tif', False, False)
        cameras = []
        cam_id = 0
        for file in filenames:
            Img, info = tf.loadTIF(file, 0, False, False)
            cameras.append(Img)
            cam_id = cam_id + 1
        cameras = np.array(cameras)
        pattern.append(cameras)
    pattern = np.array(pattern)

    imgpoints = []
    for i in tf.tqdm(range(0, pattern.shape[0]), desc='analysing pattern'):
        pattern_valid = True
        camera_corners = []
        for c in range(0, 4):
            ret, corners = cv2.findChessboardCornersSB(pattern[i, c, :, :], CHECKERBOARD, \
                                                       cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            pattern_valid = pattern_valid and ret
            camera_corners.append(corners)
        if pattern_valid:
            for c in range(0, 4):
                cv2.cornerSubPix(pattern[i, c, :, :], camera_corners[c], (11, 11), (-1, -1), criteria)
                if rotation_invariant:
                    print('pattern', i, 'camera', c)
                    camera_corners[c] = verify_img_pnts(pattern[i, c, :, :], camera_corners[c], \
                                                        CHECKERBOARD[0], camera_size_factor[c], False)
            objpoints.append(objp)
            imgpoints.append(camera_corners)

    objpoints = np.array(objpoints)
    imgpoints = np.array(imgpoints)

    # print('object points ',objpoints.shape)
    # print('image points ',imgpoints.shape)

    cret = []
    cmtx = []
    cdist = []
    crvecs = []
    ctvecs = []
    for c in tf.tqdm(range(0, 4), desc='calibrate cameras'):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints[:, c, :, :, :], [w, h], None, None)
        cret.append(ret)
        cmtx.append(mtx)
        cdist.append(dist)
        crvecs.append(rvecs)
        ctvecs.append(tvecs)

    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cRot = np.zeros((6, 3, 3))
    cTrans = np.zeros((6, 3))

    cdi = 0
    for camera_pair in tf.tqdm(camera_pairs, desc='stereo calibration of camera pairs'):
        Rot = np.zeros((3, 3))
        Trans = np.zeros((3, 3))
        Essential = np.zeros((3, 3))
        Fundamental = np.zeros((3, 3))

        ret_stereo, Mat_Cam_A, dist_A, Mat_Cam_B, dist_B, Rot, Trans, Essential, Fundamental \
            = cv2.stereoCalibrate(objpoints, imgpoints[:, camera_pair[0], :, :, :],
                                  imgpoints[:, camera_pair[1], :, :, :], \
                                  cmtx[camera_pair[0]], cdist[camera_pair[0]], cmtx[camera_pair[1]],
                                  cdist[camera_pair[1]], \
                                  (w, h), Rot, Trans, Essential, Fundamental, cv2.CALIB_FIX_INTRINSIC, criteria_stereo)

        cRot[cdi, :, :] = Rot
        cTrans[cdi, :] = np.squeeze(Trans)
        cdi = cdi + 1
    if Verbose:
        show_calibration(imgpoints, pattern)
    return cmtx, cRot, cTrans, camera_pairs
######################################################################
# Helper functions
######################################################################
# Optional : rotate the calibration patterns to correct orientation.
# If regular patterns are used, do not use this function!
def get_reference(I, sf, Verbose = False):
    center = []
    xaxis = []
    yaxis = []

    I = img_as_float(I) * 255
    # kernel = np.ones((2,2),np.float32)/2/2
    # I = cv2.filter2D(I,-1,kernel)
    msk = I < 125
    if sf == 1:
        msk = morphology.binary_closing(msk, np.ones((3, 3)))
    else:
        msk = morphology.binary_closing(msk, np.ones((5, 5)))
    label_image = label(msk)
    tab = measure.regionprops_table(label_image, properties=['label', 'area'])

    id_marker = np.argsort(tab['area'])[::-1][2]
    # print(tab['label'][id_marker])
    label_image = label_image == tab['label'][id_marker]
    dt_image = distance_transform_edt(label_image)

    threshold = 0
    count = 0
    while count < 3:
        dt_labelled = dt_image * (dt_image > threshold)
        tab = measure.regionprops_table(label(dt_labelled > 0), properties=['label', 'area', 'centroid'])
        count = tab['area'].shape[0]
        threshold = threshold + 1

    dt_labelled = dt_image * (dt_image > threshold)
    dt_labelled = morphology.binary_opening(dt_labelled, np.ones((2, 2)))

    tab = measure.regionprops_table(label(dt_labelled > 0), properties=['label', 'area', 'centroid'])
    id_pnts = np.argsort(tab['area'])[::-1]

    pnts = np.stack([tab['centroid-1'][id_pnts], tab['centroid-0'][id_pnts]], axis=1)
    # print(pnts)

    if (Verbose):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(321)
        ax.imshow(I, cmap='Greys_r')
        ax = plt.subplot(322)
        ax.imshow(msk, cmap='Greys_r')
        ax = plt.subplot(323)
        ax.imshow(label_image)
        ax = plt.subplot(324)
        ax.imshow(dt_image)
        ax = plt.subplot(325)
        ax.imshow(label(label(dt_labelled > 0)))
        ax = plt.subplot(326)
        ax.imshow(I, cmap='Greys_r')
        ax.arrow(pnts[0][0], pnts[0][1], pnts[1][0] - pnts[0][0], pnts[1][1] - pnts[0][1], \
                 color='r', width=5, length_includes_head=True)
        ax.arrow(pnts[0][0], pnts[0][1], pnts[2][0] - pnts[0][0], pnts[2][1] - pnts[0][1], \
                 color='g', width=5, length_includes_head=True)
        # ax.plot(centroid[0],centroid[1],'r*')
        # ax.plot([centroid[0]-v[0,0]*100,centroid[0]+v[0,0]*100],[centroid[1]-v[1,0]*100,centroid[1]+v[1,0]*100],'r-')

    return pnts

# Optional : Calculate x and y axes of the pattern based on the identified centers of the special calibration symbol
# If regular patterns are used, do not use this function!
def get_coords(pnt,coord,h):
    center = coord[0]
    xaxis = np.array([coord[1][0]-coord[0][0],coord[1][1]-coord[0][1]])
    xaxis = xaxis / np.linalg.norm(xaxis)
    yaxis = np.array([coord[2][0]-coord[0][0],coord[2][1]-coord[0][1]])
    yaxis = yaxis / np.linalg.norm(yaxis)
    return [np.dot((pnt-center),xaxis),np.dot((pnt-center),yaxis)]

# Optional : Renumber image points if patterns calibration patterns were flipped.
def verify_img_pnts(I, pnts, grid_c, sf, Verbose=False):
    coord = get_reference(I, sf, Verbose)
    if Verbose:
        fig = plt.figure(figsize=(19, 7))
        ax = plt.subplot(121)
        ax.imshow(I, cmap='Greys_r')

    positions_in_ref_coord = []
    i = 0
    for pnt in pnts:
        if Verbose:
            ax.annotate(str(i), (pnt[0][0], pnt[0][1]), color='r', fontsize=9)
        # print(i,get_coords(np.squeeze(pnt),coord,I.shape[0]))
        i = i + 1
        positions_in_ref_coord.append(get_coords(np.squeeze(pnt), coord, I.shape[0]))

    ids = np.asarray((np.linspace(0, len(pnts) - 1, len(pnts))), dtype='uint8')
    ids_mat = np.reshape(np.copy(ids), (int(ids.shape[0] / grid_c), int(grid_c)))

    # print(ids_mat)
    # print(positions_in_ref_coord)

    if (positions_in_ref_coord[0][0] > positions_in_ref_coord[grid_c - 1][0]):  # flip horizontal
        ids_mat = np.fliplr(ids_mat)

    if (positions_in_ref_coord[0][1] < positions_in_ref_coord[-1][1]):  # flip vertical
        ids_mat = np.flipud(ids_mat)

    # print(ids_mat)
    pnts = pnts[ids_mat.flatten(), :, :];

    if Verbose:
        ax = plt.subplot(122)
        ax.imshow(I, cmap='Greys_r')
        i = 0
        for pnt in pnts:
            ax.annotate(str(i), (pnt[0][0], pnt[0][1]), color='g', fontsize=9)
            # print(i,get_coords(np.squeeze(pnt),coord,I.shape[0]))
            i = i + 1

    return pnts



def show_calibration(imgpoints, pattern):
    pattern_id = 0
    fig = plt.figure(figsize=(10, 7), facecolor='k')
    gs = gridspec.GridSpec(2, 2, wspace=.2, hspace=.2)
    axl = []
    for cam in range(4):
        ax = plt.subplot(gs[cam])
        img = pattern[pattern_id][cam]
        ax.imshow(img, cmap='Greys_r')
        ax.axis('off')
        axl.append(ax)

    print(imgpoints.shape)
    print(pattern.shape)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=len(camera_pairs) - 1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='Set1')

    i = 0
    for camera_pair in camera_pairs:
        for pnts in range(imgpoints.shape[2]):
            # print(imgpoints[pattern_id,camera_pair[0],pnts,0,:])
            if (pnts < 3):
                con = ConnectionPatch(xyA=imgpoints[pattern_id, camera_pair[0], pnts, 0, :], \
                                      xyB=imgpoints[pattern_id, camera_pair[1], pnts, 0, :], \
                                      coordsA="data", coordsB="data", \
                                      axesA=axl[camera_pair[0]], axesB=axl[camera_pair[1]], color=mapper.to_rgba(i))
                axl[camera_pair[1]].add_artist(con)
        i = i + 1
    plt.show()
    return
