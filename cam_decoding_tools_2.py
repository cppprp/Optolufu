import tool_functions as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import flood
from skimage.morphology import binary_erosion
from skimage import measure, filters
from sklearn.neighbors import NearestNeighbors
import cv2
import paths_and_flags as pf
from skimage.segmentation import clear_border
import plotly.graph_objects as go
import pickle
##############################################################################
# how many encoding cycles
repeats = 3
# how many bit planes used
encoding_planes = 13
# how many frames per one single bit plane 'projection time'
frames_per_encoding_plane = 10
# deal with different magnification of the 4 different cameras
camera_size_factors = [1, 1, 2, 2]
# size for neighbourhood check
neighborhood_size = 6
# encoding line length
encoding_line_length = 63


######################################################################
# 4 cam
def decode_track_solve_all(mouse, cmtx, cRot, cTrans, cam_number, Verbose=False):
    # data_list = []

    # generate matrix in DLT style
    PA = cmtx[0] @ np.concatenate([cRot[0, :, :], np.expand_dims(cTrans[0], axis=1)], axis=-1)
    PB = cmtx[1] @ np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    PC = cmtx[2] @ np.concatenate([cRot[1, :, :], np.expand_dims(cTrans[1], axis=1)], axis=-1)
    PD = cmtx[3] @ np.concatenate([cRot[2, :, :], np.expand_dims(cTrans[2], axis=1)], axis=-1)
    P = np.stack((PB, PD, PA, PC), axis=2)

    print('\ntracking mouse ...', mouse, '\n')
    streams = []
    #streams.append(str(mouse) + '/cameraB.TIF')
    #streams.append(str(mouse) + '/cameraA.TIF')
    streams.append(str(mouse) + '/cameraB.TIF')  # reference camera goes first
    streams.append(str(mouse) + '/cameraD.TIF')
    streams.append(str(mouse) + '/cameraA.TIF')
    streams.append(str(mouse) + '/cameraC.TIF')
    codes_found = []
    image_data = []
    decode_list = []
    mapped_pnts = []
    index_map = []

    print('\nstart decoding process ...')
    for c in range(0, cam_number):
        # analyze stream

        E, st, en = analyze_stream(streams[c], Verbose, 'mouse:' + mouse + ' stream:' + streams[c])
        # load stream data
        I = io.imread(streams[c])[st:en, :, :]

        # print(np.percentile(S, 90))
        # print("Threshold: ", np.max(S) * 0.45)
        val = filters.threshold_otsu(I)
        print(f"Threshold: {val}")
        S = I > (val-10)
        S = S.astype(dtype='uint8')
        image_data.append(S)

        # mask name
        directory, filename = os.path.split(streams[c])
        msk_filename = filename.replace(".TIF", "_msk.TIF")
        msk_filepath = os.path.join(directory, msk_filename)
        crp_ind = int((en - st) / 2)
        msk = None
        if os.path.exists(msk_filepath):
            msk = io.imread(msk_filepath) > 0
        else:
            print(image_data[c][0].shape)
            polygon_drawer = tf.PolygonDrawerMatplotlib(image_data[c][0])
            msk = polygon_drawer.run()
            # save for next time
            io.imsave(msk_filepath, msk)
        # shrink the mask
        msk = binary_erosion(msk, np.ones((5, 5)))
        d = decode_alg(E, msk, verbose=False, sf=camera_size_factors[c])
        '''if Verbose:
            plt.figure(figsize=(19, 10))
            for i in range(d.shape[0]):
                plt.plot(d[i, 1], d[i, 2], 'o')
                plt.annotate(str(d[i, 0]), (d[i, 1], d[i, 2]))
            plt.show()'''
        if Verbose:
            x_vals = [item[1] for item in d]
            y_vals = [item[2] for item in d]
            text = [str(int(item[0])) for item in d]
            fig = go.Figure()
            fig.layout.autosize = False
            fig.layout.height= 1280
            fig.layout.width = 960
            fig.layout.plot_bgcolor = 'white'
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers+text",
                text=text,
                fillcolor='blue',
                textposition="top center"

            ))
            fig.show()

        # list must be duplicate free
        d = remove_duplicates(d)
        # check point conditions
        d = np.delete(d, validate_local_neighboors(d, False), axis=0)
        # data.append({'stream':streams[c],'frames':E,'start_seq':st,'end_seq':en,'mask':msk,'codes':d})
        decode_list.append(np.copy(d))

        # plt.figure(figsize=(19, 10))
        # for i in range(d.shape[0]):
        #   plt.plot(d[i, 1], d[i, 2], '*')
        #   plt.annotate(str(d[i, 0]), (d[i, 1], d[i, 2]))
        # plt.show()
        for code in d[:, 0]:
            pos = index_of(int(code), codes_found)
            if pos == -1:
                codes_found.append([int(code), 0, pow(2, c)])
            else:
                codes_found[pos][1] = codes_found[pos][1] + 1
                codes_found[pos][2] = codes_found[pos][2] | pow(2, c)

    codes_found = np.array(codes_found)
    print(codes_found)

    print(len(codes_found[codes_found[:, 1] > 0]), 'codes found in at least two cameras')

    # crop amount of frames
    length = image_data[0].shape[0]
    for cr in range(0, cam_number):
        length = min(length, image_data[cr].shape[0])
    for cp in range(0, cam_number):
        image_data[cp] = image_data[cp][:length, :, :]

    print('\nstart tracking process ...')
    # Tracking starts
    if cam_number == 2:
        multitutde = (1,)
    else:
        multitutde = (1, 2, 3, 4)


    data_structure = {}
    irreg = []
    for mult in multitutde:
        mult_codes = codes_found[codes_found[:, 1] == mult]
        print('\nAnalysing points with a multitude of ' + str(mult + 1))

        for mcode_id in tf.tqdm(range(mult_codes.shape[0])):
            # we need the point to be visible in camB -> bit 1 must be set
            if mult_codes[mcode_id, 2] & 0b0010 > 0:
                if mcode_id not in data_structure:
                    data_structure[mcode_id] = {}
                DLT_P = []
                DLT_Tracks = []
                for cam in range(0, cam_number):
                    # print('camera_id',c)
                    # print('code',mult_codes[mcode_id,:])

                    if mult_codes[mcode_id, 2] & pow(2, cam) > 0:
                        # generate matrix
                        DLT_P.append(P[:, :, cam].flatten())
                        # track in camera
                        code_cam_id = np.where(decode_list[cam][:, 0] == mult_codes[mcode_id, 0])[0]
                        lastp = np.squeeze(np.copy(decode_list[cam][code_cam_id, 1:3]))
                        track = tracking2(image_data[cam], image_data[cam].shape[0], lastp)
                        #irregularity = np.std(np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1)))
                        irregularity = np.sqrt((track[0][0]-track[1480][0])**2+(track[0][1]-track[1480][1])**2)
                        if (irregularity>20):
                            print (f"code ID {mult_codes[mcode_id, 0]}, irreg: {irregularity}")
                            irreg.append(mcode_id)
                            bad_tracks(image_data[cam], image_data[cam].shape[0], lastp, pf.target_path+f"/temp/{mult_codes[mcode_id, 0]}/")
                        DLT_Tracks.append(track)
                        data_structure[mcode_id][cam] = track
                DLT_Tracks = np.array(DLT_Tracks)
                # print(DLT_Tracks.shape)
                # perform DLT
                conv_pos = np.zeros((3, DLT_Tracks.shape[1]))
                for frame_id in range(DLT_Tracks.shape[1]):
                    conv_pos[:, frame_id] = DLTrecon(3, mult + 1, np.array(DLT_P),
                                                     np.squeeze(DLT_Tracks[:, frame_id, :]))
                    # add to converted pnt and add pnt id
                mapped_pnts.append(conv_pos)
                index_map.append(mult_codes[mcode_id, 0])

    # data_list.append({'mouse': mouse, 'mapped_pnts': mapped_pnts, 'index_map': index_map})
    draw_correspondence_t(data_structure, image_data, pf.target_path+'/temp/'+str(mouse)+'/')
    return np.asarray(mapped_pnts), index_map


######################################################################
# Helper functions
######################################################################
# detect intersection with the level function
def intersection_pos(v, lvl):
    I = []
    for i in range(1, len(v)):
        if (v[i - 1] < lvl and v[i] > lvl):
            I.append(i)
        if (v[i - 1] > lvl and v[i] < lvl):
            I.append(i - 1)

    if (v[len(v) - 1] > lvl):  # light till end of acquisition
        I.append(len(v) - 1)

    return I


# analyze camera stream and identify encoding and measurement phases
def phases(I, verbose=False):
    v = np.average(I, axis=(1, 2))
    lvl = np.max(v) * .9
    #lvl = filters.threshold_otsu(v)-5.0
    mi = np.min(v)
    ind = np.argwhere(v>lvl)
    encode = []
    start = int(ind[0])
    stop = 0
    for i in range(0, repeats):
        stop = int(start + frames_per_encoding_plane*encoding_planes)
        step = int(np.floor(frames_per_encoding_plane/2))
        encoding_vec = list(range(start+step, stop+1, frames_per_encoding_plane))
        start = stop+frames_per_encoding_plane # extra frames account for the dark phase
        encode.append(encoding_vec)
    return encode, start, ind[-1]


# analyse one camera
def analyze_stream(camA_stream, verbose=False, title=''):
    I = io.imread(camA_stream)
    #I = I[0:3000, :,:] # only for this one plethysmograohy run, delete me in the future
    print('shape ',I.shape)
    # plt.figure()
    # plt.imshow(I[0,150:250,200:300])
    # plt.show()
    enc, start, end = phases(I, verbose)
    enc = np.array(enc, dtype='int')
    # print(enc)
    E = []
    for e in range(0, enc[0].shape[0]):
        I_stk = []
        for r in range(0, repeats):
            I_stk.append(I[enc[r, e], :, :])
        # E.append(np.median(np.array(I_stk),axis=0))
        E.append(np.median(np.array(I_stk), axis=0))

    E = np.array(E)
    if verbose:
        fig = plt.figure(figsize=(19, 10))
        s = int(np.ceil(np.sqrt(len(enc[0]))))

        for i in range(0, len(enc[0])):
            ax = plt.subplot(s, s, i + 1)
            ax.imshow(E[i, :, :], vmin=0, vmax=255)
            ax.axis('off')
            ax.set_title('encoding ' + str(i))
        fig.suptitle(title)
        plt.show()

    return E, int(start), int(end)


# decoding algorithm
def decode_alg(E, msk, verbose=False, sf=1):
    THRESHOLD_BIT = 0.20
    MAX_AREA_MULTIPLIER = 150
    MIN_AREA_BASE = 5
    #MIN_AREA_MULTIPLIER = 5
    search_roi = 6
    half_window = int(search_roi * sf / 2)

    range_max = np.max(E)

    # find positions in static frame
    static_pattern = E[0, :, :] * msk
    upper_threshold, lower_threshold = tf.live_thresholding(static_pattern)
    #val = filters.threshold_otsu(static_pattern[static_pattern > 20])
    #print(f"Threshold here: {lower_threshold}")


    # Threshold the stream
    static_pattern = static_pattern>lower_threshold
    static_pattern = static_pattern.astype(dtype='uint8')

    #inv_msk = ~msk
    #static_filtered = inv_msk & static_pattern
    '''msk[0, :] = 0
    msk[-1, :] = 0
    msk[:, 0] = 0
    msk[:, -1] = 0
'''
    blobs_labels = measure.label(static_pattern, background=0)
    blobs_labels = clear_border(blobs_labels, mask=msk)

    # remove labels outside size range
    tab = measure.regionprops_table(blobs_labels, properties=['label', 'area'])
    for i in range(0, len(tab['label'])):
        if tab['area'][i] < MIN_AREA_BASE * sf or tab['area'][i] > MAX_AREA_MULTIPLIER * sf:
            blobs_labels = blobs_labels - (blobs_labels == tab['label'][i]) * tab['label'][i]

    # final pass get centroid
    tab = measure.regionprops_table(blobs_labels, properties=['label', 'centroid'])

    # cpy into vector
    decode = np.zeros((len(tab['label']), 3))
    decode[:, 0] = 0
    decode[:, 1] = tab['centroid-1'] #x coord
    decode[:, 2] = tab['centroid-0'] #y coord

    pos_track = np.zeros((len(tab['label']), encoding_planes))
    for i in range(0, decode.shape[0]):
        # for i in range(0,1):
        #if verbose:
            #plt.figure(figsize=(19, 3))
        for f in range(1, encoding_planes):  # we remove the static pattern at the beginning
            # get frame
            F = E[f, :, :] * msk
            range_max = np.max(F)
            otsu_threshold = filters.threshold_otsu(F)


            y_start = max(0, int(decode[i, 2]) - half_window)
            y_end = min(F.shape[0], int(decode[i, 2]) + half_window)

            x_start = max(0, int(decode[i, 1]) - half_window)
            x_end = min(F.shape[1], int(decode[i, 1]) + half_window)

            # Get the cropped region
            region = F[y_start:y_end, x_start:x_end]

            # Compute the Otsu threshold for the region

            #otsu_threshold = filters.threshold_otsu(region)

            # Calculate the number of pixels above the Otsu threshold
            above_threshold_count = np.sum(region > otsu_threshold)

            # Calculate the total number of pixels in the region
            total_pixels = region.size

            # Calculate the percentage of pixels above the Otsu threshold
            percentage_above_threshold = (above_threshold_count / total_pixels) * 100

            # Set the bit to True if the percentage is above 50%
            bit = percentage_above_threshold > 20

            # Update the decode matrix
            decode[i, 0] = decode[i, 0] + bit * 2 ** (f - 1)

            if verbose:
                ax = plt.subplot(1, encoding_planes, encoding_planes - f)
                ax.imshow(F, vmin=0, vmax=255)
                ax.plot(decode[i, 1], decode[i, 2], 'r*', markersize=1)
                ax.set_title(str(int(bit)))
                ax.axis(False)
        if verbose:
            plt.suptitle(str(int(decode[i, 0])))
            plt.show()

    if verbose:
        '''plt.figure(figsize=(19, 10))
        ax = plt.subplot(221)
        ax.imshow(static_pattern)
        plt.gca().set_title('static frame')

        for i in range(0, decode.shape[0]):
            ax.plot(decode[i, 1], decode[i, 2], '+', markersize=3)
            ax.annotate(str(int(decode[i, 0])), (decode[i, 1], decode[i, 2]), color='red', fontsize=12)

        ax = plt.subplot(222)
        ax.imshow(blobs_labels)
        ax = plt.subplot(223)'''

    plt.close('all')
    return decode


# converts ids to row(r) and column(c) coordinates
def convert_id_to_r_and_c(codes):
    rc_lst = []
    try:
        for i in range(0, codes.shape[0]):
            r = int(np.floor(codes[i] / encoding_line_length))
            c = int(codes[i] % encoding_line_length)
            rc_lst.append([r, c])
    except:
        r = int(np.floor(codes / encoding_line_length))
        c = int(codes % encoding_line_length)
        rc_lst.append([r, c])
    return np.array(rc_lst)


# check from the local neighbours if the code is plausable
def validate_local_neighboors(decoded, verbose=False):
    kill_lst = []
    decoded = np.array(decoded)
    NN = NearestNeighbors(n_neighbors=neighborhood_size)
    NN.fit(decoded[:, 1:])

    if verbose:
        plt.figure(figsize=(15, 10))

    for p in range(0, decoded.shape[0]):
        knn = np.squeeze(NN.kneighbors([decoded[p, 1:]])[1])
        converted = np.squeeze(convert_id_to_r_and_c(decoded[p, 0]))
        # print(converted)
        lst = convert_id_to_r_and_c(decoded[knn, 0])
        r = converted[0]
        c = converted[1]
        counts_of_difference_greater_two = np.max([np.sum(abs(lst[:, 0] - r) > 2), np.sum(abs(lst[:, 1] - c) > 2)])

        # print(counts_of_difference_greater_two, convert_id_to_r_and_c(decoded[p,0]),'->',convert_id_to_r_and_c(decoded[knn,0]))
        if counts_of_difference_greater_two > neighborhood_size / 2:
            kill_lst.append(p)

        if verbose:
            if (counts_of_difference_greater_two > neighborhood_size / 2):
                plt.plot(decoded[p][1], decoded[p][2], 'r*')
            else:
                plt.plot(decoded[p][1], decoded[p][2], 'go')

    if verbose:
        plt.show()

    return kill_lst


# helper function
def index_of(pattern, codes):
    pos = -1
    for i in range(len(codes)):
        if codes[i][0] == pattern:
            pos = i
    return pos
def bad_tracks (S, max_frames, lastp, output_dir):
    print("Outputting visual for bad track")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Setup trace array with the first point as the init
    trace = np.zeros((max_frames, 2))
    trace[0, 0] = lastp[0]
    trace[0, 1] = lastp[1]
    # Setup velocity tracking for modified ROI generation
    velocities = np.zeros((max_frames, 2))
    old_frame = S[0]
    max_velocity_change = 10.0
    roi_size = 10
    for frame_id in range(1, max_frames):
        # Last point on previous frame
        p0x = int(trace[frame_id-1][0])
        p0y = int(trace[frame_id-1][1])

        # Dynamic ROI based on predicted movement
        # Last velocity as predictor
        #roi_size += int(np.linalg.norm(velocities[frame_id - 1]))
        swr, ewr = max(0, p0y - roi_size), min(S.shape[1] - 1, p0y + roi_size)
        swc, ewc = max(0, p0x - roi_size), min(S.shape[2] - 1, p0x + roi_size)

        # Define the ROI in the frame
        roi_frame = S[frame_id, swr:ewr, swc:ewc]

        # Grow under last centroid within the dynamic ROI
        RG = flood(roi_frame, (int(p0y - swr), int(p0x - swc)))  # adjust with ROI starting point
        tab = measure.regionprops_table(RG.astype(dtype='uint8'), properties=['label', 'centroid', 'area'])
        p1x = tab['centroid-1'][0] + swc
        p1y = tab['centroid-0'][0] + swr

        output = roi_frame * RG
        output = output.astype("uint8")*255
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
        cv2.circle(output, (int(p1x - swc), int(p1y - swr)), 1, (0, 255, 0), -1)  # Draw point for cam1
        #cv2.putText(output, f"x: {p1x}, y: {p1y}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX,0.05, (0, 255, 0), 2, cv2.LINE_AA)
        filename = os.path.join(output_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(filename, output)

        # Check point velocity change
        current_velocity = np.array([p1x, p1y]) - np.array([p0x, p0y])
        if np.linalg.norm(current_velocity) > max_velocity_change:
            # Predict point movement based on last known position
            trace[frame_id] = trace[frame_id - 1] + velocities[frame_id - 1]
            print (f"Frame: {frame_id}, pnt_x, y:{trace[0,0], trace[0,1]}")
            velocities[frame_id] = velocities[frame_id - 1]  # Store the previous velocity as the current frame's velocity
        else:
            trace[frame_id] = [p1x, p1y]
            velocities[frame_id] = np.array(current_velocity)
    return trace

def tracking2(S, max_frames, lastp):
    # Setup trace array with the first point as the init
    trace = np.zeros((max_frames, 2))
    trace[0, 0] = lastp[0]
    trace[0, 1] = lastp[1]
    # Setup velocity tracking for modified ROI generation
    velocities = np.zeros((max_frames, 2))
    old_frame = S[0]
    max_velocity_change = 10.0
    roi_size = 10
    for frame_id in range(1, max_frames):
        # Last point on previous frame
        p0x = int(trace[frame_id-1][0])
        p0y = int(trace[frame_id-1][1])

        # Dynamic ROI based on predicted movement
        # Last velocity as predictor
        #roi_size += int(np.linalg.norm(velocities[frame_id - 1]))
        swr, ewr = max(0, p0y - roi_size), min(S.shape[1] - 1, p0y + roi_size)
        swc, ewc = max(0, p0x - roi_size), min(S.shape[2] - 1, p0x + roi_size)

        # Define the ROI in the frame
        roi_frame = S[frame_id, swr:ewr, swc:ewc]
        #If centroid has a blob under it
        if roi_frame[p0y-swr,p0x-swc] == 1: #row column
            # Grow under last centroid within the dynamic ROI
            RG = flood(roi_frame, (int(p0y - swr), int(p0x - swc)))  # adjust with ROI starting point
            tab = measure.regionprops_table(RG.astype(dtype='uint8'), properties=['label', 'centroid', 'area'])
            p1x = tab['centroid-1'][0] + swc
            p1y = tab['centroid-0'][0] + swr
            # Check point velocity change
            current_velocity = np.array([p1x, p1y]) - np.array([p0x, p0y])
            # Check the motion irregularity based on velocity
            if np.linalg.norm(current_velocity) > max_velocity_change:
                # Predict point movement based on last known position
                trace[frame_id] = trace[frame_id - 1] + velocities[frame_id - 1]
                print (f"Frame: {frame_id}, pnt_x, y:{trace[0,0], trace[0,1]}")
                velocities[frame_id] = velocities[frame_id - 1]  # Store the previous velocity as the current frame's velocity
            else: # Regular movement
                trace[frame_id] = [p1x, p1y]
                velocities[frame_id] = np.array(current_velocity)
        # If centroid has void under it - predict centroid on last known velocity
        else:
            #trace[frame_id] = trace[frame_id - 1] + velocities[frame_id - 1]
            trace[frame_id] = trace[frame_id - 1]
            velocities[frame_id] = velocities[frame_id - 1]

    return trace

def remove_duplicates(d):
    unique_list = []
    unique_d = []
    for r in range(d.shape[0]):
        if not d[r, 0] in unique_list:
            unique_list.append(d[r, 0])
            unique_d.append(d[r, :])
    return np.array(unique_d)


# discreete linear transform
def DLTrecon(nd, nc, Ls, uvs):
    '''
    Reconstruction of object point from image point(s) based on the DLT parameters.
    This code performs 2D or 3D DLT point reconstruction with any number of views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.
    Inputs:
     nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
     nc is the number of cameras (views) used.
     Ls (array type) are the camera calibration parameters of each camera
      (is the output of DLTcalib function). The Ls parameters are given as columns
      and the Ls for different cameras as rows.
     uvs are the coordinates of the point in the image 2D space of each camera.
      The coordinates of the point are given as columns and the different views as rows.
    Outputs:
     xyz: point coordinates in space
    '''

    # Convert Ls to array:
    Ls = np.asarray(Ls)
    # Check the parameters:
    if Ls.ndim == 1 and nc != 1:
        raise ValueError(
            'Number of views (%d) and number of sets of camera calibration parameters (1) are different.' % (nc))
    if Ls.ndim > 1 and nc != Ls.shape[0]:
        raise ValueError(
            'Number of views (%d) and number of sets of camera calibration parameters (%d) are different.' % (
                nc, Ls.shape[0]))
    if nd == 3 and Ls.ndim == 1:
        raise ValueError('At least two sets of camera calibration parameters are needed for 3D point reconstruction.')

    if nc == 1:  # 2D and 1 camera (view), the simplest (and fastest) case
        # One could calculate inv(H) and input that to the code to speed up things if needed.
        # (If there is only 1 camera, this transformation is all Floatcanvas2 might need)
        Hinv = np.linalg.inv(Ls.reshape(3, 3))
        # Point coordinates in space:
        xyz = np.dot(Hinv, [uvs[0], uvs[1], 1])
        xyz = xyz[0:2] / xyz[2]
    else:
        M = []
        for i in range(nc):
            L = Ls[i, :]
            u, v = uvs[i][0], uvs[i][1]  # this indexing works for both list and numpy array
            if nd == 2:
                M.append([L[0] - u * L[6], L[1] - u * L[7], L[2] - u * L[8]])
                M.append([L[3] - v * L[6], L[4] - v * L[7], L[5] - v * L[8]])
            elif nd == 3:
                M.append([L[0] - u * L[8], L[1] - u * L[9], L[2] - u * L[10], L[3] - u * L[11]])
                M.append([L[4] - v * L[8], L[5] - v * L[9], L[6] - v * L[10], L[7] - v * L[11]])

        # Find the xyz coordinates:
        U, S, Vh = np.linalg.svd(np.asarray(M))
        # Point coordinates in space:
        xyz = Vh[-1, 0:-1] / Vh[-1, -1]

    return xyz

def draw_correspondence_t(data_tracks, image_data, output_dir):
    # Create an output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine video properties
    height, width = image_data[0][0].shape
    out_width = 2 * width

    # For each frame
    for frame_id in range(len(image_data[0])):
        # Create an empty frame with both camera views side by side
        frame = np.zeros((height, out_width))
        frame[:, :width] = image_data[0][frame_id]
        frame[:, width:] = image_data[1][frame_id]
        frame = frame.astype("uint8")*255
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # Loop over all the mcodes
        for mcode_id, tracks in data_tracks.items():
            if len(tracks) > 1:  # Ensure we have data for both cameras
                point_cam1 = tracks[0][frame_id]
                point_cam2 = tracks[1][frame_id]
                point_cam2 = (point_cam2[0] + width, point_cam2[1])  # Adjust x coordinate for side by side view

                cv2.circle(frame, (int(point_cam1[0]),int(point_cam1[1])) , 3, (0, 255, 0), -1)  # Draw point for cam1
                cv2.circle(frame, (int(point_cam2[0]),int(point_cam2[1])) , 3, (0, 0, 255), -1)  # Draw point for cam2

                # Draw line between the points
                cv2.line(frame, (int(point_cam1[0]),int(point_cam1[1])), (int(point_cam2[0]),int(point_cam2[1])), (255, 255, 255), 1)

        # Save the frame as an image
        filename = os.path.join(output_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(filename, frame)


