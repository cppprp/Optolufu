import tool_functions as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import flood
from skimage.morphology import binary_erosion
from skimage import measure
from sklearn.neighbors import NearestNeighbors

##############################################################################
# how many encoding cycles
repeats = 3
# how many bit planes used
encoding_planes = 13
# how many frames per one single bit plane 'projection time'
frames_per_encoding_plane = 10
# deal with different magnification of the 4 different cameras
camera_size_factors = [2, 1, 2, 1]
# size for neighborhood check
neighborhood_size = 6
# encoding line length
encoding_line_length = 63


######################################################################
# 4 cam
def decode_track_solve_all(mice, cmtx, cRot, cTrans, Verbose=False):
    data_list = []

    # generate matrix in DLT style
    PA = cmtx[0] @ np.concatenate([cRot[0, :, :], np.expand_dims(cTrans[0], axis=1)], axis=-1)
    PB = cmtx[1] @ np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    PC = cmtx[2] @ np.concatenate([cRot[1, :, :], np.expand_dims(cTrans[1], axis=1)], axis=-1)
    PD = cmtx[3] @ np.concatenate([cRot[2, :, :], np.expand_dims(cTrans[2], axis=1)], axis=-1)
    P = np.stack((PA, PB, PC, PD), axis=2)

    for mouse in tf.tqdm(mice, 'analysed mice'):

        print('\nanalyzing ...', mouse, '\n')
        streams = np.sort(tf.getListOfFiles(str(mouse), 'camera..tif', False, False))
        codes_found = []
        image_data = []
        decode_list = []
        # mapped_pnts[marker,:,frame];
        mapped_pnts = []
        index_map = []

        print('\nstart decoding process ...')
        for c in range(0, len(streams)):
            # analyze stream
            E, st, en = analyze_stream(streams[c], Verbose, 'mouse:' + mouse + ' stream:' + streams[c])
            # load stream data
            I = io.imread(streams[c])[st:en, :, :]
            S = I > np.max(I) * .5
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
                print (image_data[0][0].shape)
                polygon_drawer = tf.PolygonDrawerMatplotlib(image_data[0][0])
                msk = polygon_drawer.run()
                #save for next time
                io.imsave(msk_filepath, msk)
            # shrink the mask
            msk = binary_erosion(msk, np.ones((5, 5)))
            d = decode_alg(E, msk, None, False, camera_size_factors[c])
            if Verbose:
                plt.figure(figsize=(19, 10))
                for i in range(d.shape[0]):
                    plt.plot(d[i, 1], d[i, 2], 'o')
                    plt.annotate(str(d[i, 0]), (d[i, 1], d[i, 2]))
                plt.show()

            # list must be duplicate free
            d = remove_duplicates(d)
            # check point conditions
            d = np.delete(d, validate_local_neighboors(d, False), axis=0)
            # data.append({'stream':streams[c],'frames':E,'start_seq':st,'end_seq':en,'mask':msk,'codes':d})
            decode_list.append(np.copy(d))

            #plt.figure(figsize=(19, 10))
            #for i in range(d.shape[0]):
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
        for c in range(0, len(streams)):
            length = min(length, image_data[c].shape[0])
        for c in range(0, len(streams)):
            image_data[c] = image_data[c][:length, :, :]

        print('\nstart tracking process ...')
        for mult in range(1, 4):
            mult_codes = codes_found[codes_found[:, 1] == mult]
            print('\nAnalysing points with a multitude of ' + str(mult + 1))
            for mcode_id in tf.tqdm(range(mult_codes.shape[0])):
                # we need the point to be visible in camB -> bit 1 must be set
                if mult_codes[mcode_id, 2] & 0b0010 > 0:
                    DLT_P = []
                    DLT_Tracks = []
                    for c in range(4):
                        # print('camera_id',c)
                        # print('code',mult_codes[mcode_id,:])
                        if mult_codes[mcode_id, 2] & pow(2, c) > 0:
                            # generate matrix
                            DLT_P.append(P[:, :, c].flatten())
                            # track in camera
                            code_cam_id = np.where(decode_list[c][:, 0] == mult_codes[mcode_id, 0])[0]
                            lastp = np.squeeze(np.copy(decode_list[c][code_cam_id, 1:3]))
                            DLT_Tracks.append(tracking(image_data[c], image_data[c].shape[0], lastp))
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

        data_list.append({'mouse': mouse, 'mapped_pnts': mapped_pnts, 'index_map': index_map})
    return data_list

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
    mi = np.min(v)
    ipos = intersection_pos(v, lvl)
    if verbose:
        print('intersection positions = ', ipos)
    pos = 0
    encode = []

    # startpos_encoding = int((ipos[0]+ipos[1])/2)
    encoding_inc = frames_per_encoding_plane

    for i in range(0, repeats):
        startpos_encoding = ipos[pos] + 4
        encoding_vec = np.linspace(0, encoding_planes - 1, encoding_planes) * encoding_inc + startpos_encoding
        encoding_vec = encoding_vec.astype(dtype='int')
        encode.append(encoding_vec)
        pos = pos + 4

    if verbose:
        plt.figure(figsize=(19, 5))
        plt.plot(np.linspace(0, len(v[:ipos[-2]]) - 1, len(v[:ipos[-2]])), v[:ipos[-2]], 'b-')
        plt.plot([0, len(v[:ipos[-2]]) - 1], [lvl, lvl], 'r-')
        # plt.plot(np.linspace(0,len(v)-1,len(v)),v,'b-')
        # plt.plot([0,len(v)-1],[lvl,lvl],'r-')
        # print('positions ',ipos)
        for p in ipos[:len(ipos) - 1]:
            plt.plot(p, v[p], 'r*', markersize=12)
        for encoding_phase in encode:
            for e in encoding_phase:
                plt.plot([e, e], [mi, lvl * 1.1], 'g-')

    return encode, encoding_inc, ipos[-2], ipos[-1]


# analyse one camera
def analyze_stream(camA_stream, verbose=False, title=''):
    I = io.imread(camA_stream)
    # print('shape ',I.shape)
    # plt.figure()
    # plt.imshow(I[0,150:250,200:300])
    # plt.show()
    enc, inc, start, end = phases(I, verbose)
    enc = np.array(enc)
    # print(enc)
    E = []
    for e in range(0, enc[0].shape[0]):
        I_stk = []
        for r in range(0, repeats):
            I_stk.append(I[enc[r, e], :, :])
        # E.append(np.median(np.array(I_stk),axis=0))
        E.append(np.average(np.array(I_stk), axis=0))

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

    return E, start, end


# decoding algorithm
def decode_alg(E, msk, roi=None, verbose=False, sf=1):
    try:
        if msk == None:
            msk = np.ones((E.shape[1], E.shape[2]))
    except:
        pass
    try:
        if roi == None:
            roi = [0, E.shape[1], 0, E.shape[2]]
    except:
        pass

    # unsharpen
    # kernel = np.ones((sf*2+1,sf*2+1),np.float32)/25
    # for i in range(0,E.shape[0]):
    #    E[i,:,:] = cv2.filter2D(E[i,:,:],-1,kernel)

    range_min = np.min(E)
    range_max = np.max(E)

    # find positions ins static frame
    S = E[0, :, :] * msk
    S = S[roi[0]:roi[1], roi[2]:roi[3]]
    omsk = np.copy(msk)
    msk = msk[roi[0]:roi[1], roi[2]:roi[3]]

    ma = np.max(S)
    S = S > ma * .5
    # S = img_as_ubyte(S)
    S = S.astype(dtype='uint8')

    for i in range(0, msk.shape[0]):
        msk[i, 0] = 0
        msk[i, msk.shape[1] - 1] = 0
    for i in range(0, msk.shape[1]):
        msk[0, i] = 0
        msk[msk.shape[0] - 1, i] = 0
    msk = 1 - msk

    # MD = cv2.distanceTransform(S, cv2.DIST_L2, 3) > 1
    # 1st pass
    # blobs_labels = measure.label(MD, background=0)
    blobs_labels = measure.label(S, background=0)
    # remove labels touching edge of mask
    edge = blobs_labels * msk
    tab = measure.regionprops_table(edge.astype(dtype='uint8'))
    for l in tab['label']:
        blobs_labels = blobs_labels - (blobs_labels == l) * l
    # remove labels outside size range
    tab = measure.regionprops_table(blobs_labels, properties=['label', 'area'])
    for i in range(0, len(tab['label'])):
        if tab['area'][i] < 5 + (sf - 1) * 15 or tab['area'][i] > 100 * sf:
            blobs_labels = blobs_labels - (blobs_labels == tab['label'][i]) * tab['label'][i]

    # final pass get centroid
    tab = measure.regionprops_table(blobs_labels, properties=['label', 'centroid'])

    # cpy into vector
    decode = np.zeros((len(tab['label']), 3))
    decode[:, 0] = 0
    decode[:, 1] = tab['centroid-1']
    decode[:, 2] = tab['centroid-0']

    for i in range(0, decode.shape[0]):
        # for i in range(0,1):
        if verbose:
            plt.figure(figsize=(19, 3))
        for f in range(0, encoding_planes - 1):  # we remove the static pattern at the beginning
            # get frame
            F = E[f + 1, :, :] * omsk
            F = F[roi[0]:roi[1], roi[2]:roi[3]]

            # ma = np.max(F)
            # F = F>ma*.8
            # F = F.astype(dtype='uint8')
            # F = cv2.distanceTransform(F, cv2.DIST_L2, 3) > 1

            bit = np.average(F[max(int(decode[i, 2]) - (4 - sf), 0):min(F.shape[0], int(decode[i, 2]) + 3 + sf), \
                             max(0, int(decode[i, 1]) - (4 - sf)):min(F.shape[1],
                                                                      int(decode[i, 1]) + 3 + sf)]) > range_max * .2

            decode[i, 0] = decode[i, 0] + bit * 2 ** f

            if verbose:
                ax = plt.subplot(1, encoding_planes, encoding_planes - f)
                ax.imshow(F, vmin=0, vmax=255)
                ax.plot(decode[i, 1], decode[i, 2], 'r*', markersize=1)
                ax.set_title(str(int(bit)))
                ax.axis(False)

            #
            # if bit>0:
            #    RG = flood(F,(int(decode[i,3]),int(decode[i,2])),tolerance=30)
            #    tab = measure.regionprops_table(RG.astype(dtype='uint8'),properties=['label','centroid','area'])
            #    decode[i,2]=tab['centroid-1'][0]
            #    decode[i,3]=tab['centroid-0'][0]
            # else:
            #    RG = np.zeros(F.shape)

            # ax = plt.subplot(122)
            # ax.imshow(RG)
        if verbose:
            plt.suptitle(str(int(decode[i, 0])))
            plt.show()

    if verbose:
        plt.figure(figsize=(19, 10))
        ax = plt.subplot(221)
        ax.imshow(S)
        plt.gca().set_title('static frame')

        for i in range(0, decode.shape[0]):
            ax.plot(decode[i, 1], decode[i, 2], '+', markersize=3)
            ax.annotate(str(int(decode[i, 0])), (decode[i, 1], decode[i, 2]), color='red', fontsize=12)

        ax = plt.subplot(222)
        ax.imshow(blobs_labels)
        ax = plt.subplot(223)
        ax.imshow(edge)

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


# tracking algorithm
def tracking(S, max_frames, lastp):
    trace = np.zeros((max_frames, 2))
    roi_size = 20
    stp = np.copy(lastp)

    # for frame_id in tqdm(range(max_frames),desc='tracking frame'):
    for frame_id in range(max_frames):
        swr = max(0, int(lastp[1]) - roi_size)
        ewr = min(S.shape[1] - 1, int(lastp[1]) + roi_size)
        swc = max(0, int(lastp[0]) - roi_size)
        ewc = min(S.shape[2] - 1, int(lastp[0]) + roi_size)

        # print(swr,ewr,swc,ewc)

        ROI = np.squeeze(S[frame_id, swr:ewr + 1, swc:ewc + 1])
        pos = [lastp[0] - swc, lastp[1] - swr]

        # Region growing
        RG = flood(ROI, (int(pos[0]), int(pos[1])))
        tab = measure.regionprops_table(RG.astype(dtype='uint8'), properties=['label', 'centroid', 'area'])
        lastp[0] = tab['centroid-1'][0]
        lastp[1] = tab['centroid-0'][0]

        # ax=plt.subplot(121)
        # ax.imshow(ROI,cmap='Greys_r')
        # ax.plot(pos[0],pos[1],'r*')
        # ax=plt.subplot(122)
        # ax.imshow(RG)
        # ax.plot(lastp[0],lastp[1],'r*')
        # plt.show()

        # correct for ROI pos
        lastp[0] = lastp[0] + swc
        lastp[1] = lastp[1] + swr

        trace[frame_id, 0] = lastp[0]
        trace[frame_id, 1] = lastp[1]

    # print('tracking start',stp,'end',lastp,'length',trace.shape[0])
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
