import os

import matplotlib.pyplot as plt

import paths_and_flags as pf
import tool_functions as tf
import pyvista as pv
import numpy as np
import copy

# encoding line length
encoding_line_length = 63
grid_size_cols = encoding_line_length
fps = 100
start_frame = 0

def mesh_and_render(data_list):
    for data in data_list:

        mapped_pnts = np.asarray(data['mapped_pnts'])
        L = data['index_map']
        mouse_folder = data['mouse']
        print('processing', mouse_folder)
        mouse = mouse_folder.split('/')[-1]
        ###########################
        # the KLT reorientation
        if pf.use_KLT:
            end_frame = 100
            # end_frame = Y.shape[2]-1
            print('performing KLT in the range of ', start_frame, ' to ', end_frame)

            pnts = []
            for f in range(start_frame, end_frame):
                for m in range(0, mapped_pnts.shape[0]):
                    pnts.append(mapped_pnts[m, :, f])

            pnts = np.array(pnts)

            centroid = np.mean(pnts, axis=0)
            print('centroid = ', centroid)
            S = np.cov((pnts - centroid).T)

            # calculating eigenvectors and values
            w, v = np.linalg.eig(S)

            # sort them in descending order
            order = np.flip(np.argsort(w))
            v[:, [0, 1, 2]] = v[:, [order[0], order[1], order[2]]]
            print(v)
            if (v[2, 2] > 0):
                v = v @ np.mat([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                print('new v')
                print(v)

            # apply
            end_frame = mapped_pnts.shape[2]
            for f in tf.tqdm(range(start_frame, end_frame), desc='Apply KLT ...'):
                for m in range(0, mapped_pnts.shape[0]):
                    mapped_pnts[m, :, f] = np.squeeze(np.matmul(v.T, mapped_pnts[m, :, f] - centroid))
            # get a camera coordinate baseline
            base_vec = np.squeeze(np.matmul(v.T, -centroid))
            # create a new array with baseline for later quantification
            mapped_pnts_q = copy.deepcopy(mapped_pnts)
            mapped_pnts_q[:, 0, :] = mapped_pnts[:, 0, :] + base_vec[0, 0]
            mapped_pnts_q[:, 1, :] = mapped_pnts[:, 1, :] + base_vec[0, 1]
            mapped_pnts_q[:, 2, :] = mapped_pnts[:, 2, :] + base_vec[0, 2]

        else:
            print('KLT skipped')

        current_pnt = identify_reference_pnt(mapped_pnts, L, verbose=False)
        ref_insp, ref_exp = define_insp_and_exp_frame(current_pnt, mapped_pnts)

        #close holes in mesh
        mapped_pnts_int, L_int = repair_holes(mapped_pnts, L, grid_size_cols)

        frame = ref_exp
        base_frame = ref_insp
        pf.amplified = False
        reference_mesh = polygonize_frame(base_frame, mapped_pnts_int, L_int,
                                          verbose=False,
                                          export_name=pf.target_path + '/' +
                                                      str(mouse) + '_expiration.ply', mouse=mouse)
        moved_mesh = polygonize_frame(frame, mapped_pnts_int, L_int,
                                      verbose=False,
                                      export_name=pf.target_path + '/' +
                                                  str(mouse) + '_inspiration.ply', mouse=mouse)

        distance = compute_distances(reference_mesh, moved_mesh)
        diff_mesh = visualize_difference(reference_mesh, moved_mesh, distance)
        diff_mesh.save(pf.target_path + '/' + str(mouse) + '_difference.ply')

        if pf.generate_animation:
            cmap = 'RdBu_r'
            plotter = pv.Plotter(off_screen=True)
            f_name = pf.target_path + str(mouse) + '_animation.mp4'
            plotter.open_movie(f_name, framerate=100)
            # Set up initial view (using the first mesh for reference)
            plotter.add_mesh(reference_mesh)
            plotter.add_mesh(reference_mesh.outline_corners(), show_edges=True)
            plotter.view_isometric()
            print('Orient the view, then press "q" to close window and produce movie')
            # Render and do NOT close
            plotter.show(auto_close=False)
            # Run through each frame
            for fr in tf.tqdm(range(mapped_pnts_int.shape[2])):
                mesh_at_frame = polygonize_frame(fr, mapped_pnts_int, L_int, verbose=False, mouse=mouse)
                distance = compute_distances(reference_mesh, mesh_at_frame)
                mesh_at_frame['Distances'] = distance
                plotter.add_mesh(mesh_at_frame,
                                 scalars='Distances',
                                 cmap=cmap,
                                 #clim=[-np.max(np.abs(distance)), np.max(np.abs(distance))],
                                 clim = [-1.5, 1],
                                 show_edges=True,
                                 smooth_shading=True
                                 )
                plotter.write_frame()
                plotter.clear()
            plotter.close()
        else:
            print('skipped generation of animation')


###############################################################################################################################
# add only vertexes that are not already there
def find_or_add(vertex, vertex_id_map, r, c, index_map, mapped_pnts):
    try:
        _id = vertex_id_map.index(int(r * grid_size_cols + c))
    except:
        _id = len(vertex_id_map)
        vertex_id_map.append(r * grid_size_cols + c)
        _oid = np.where(index_map == r * grid_size_cols + c)
        vertex.append((mapped_pnts[_oid, 0].flatten()[0],
                       mapped_pnts[_oid, 1].flatten()[0],
                       mapped_pnts[_oid, 2].flatten()[0]))
        # vertex.append((mapped_pnts[_oid,1],mapped_pnts[_oid,2],mapped_pnts[_oid,0]))
    return _id, vertex, vertex_id_map


def identify_reference_pnt(mapped_pnts, index_list, verbose = False):
    current_frame = int(mapped_pnts.shape[2] / 2)
    if (verbose):
        fig2, ax1 = tf.plt.subplots(dpi=150)
        for ID in range (len(index_list)):
            r = np.floor(index_list[ID] / grid_size_cols)
            c = index_list[ID] % grid_size_cols
            ax1.plot(mapped_pnts[:, 0, current_frame], mapped_pnts[:, 1, current_frame], 'k.', markersize=6)
            ax1.set_aspect('equal')
            ax1.annotate(str(r)+','+str(c), (mapped_pnts[ID, 0, current_frame] + 0.05, mapped_pnts[ID, 1, current_frame] - 0.1),
                             fontsize=6.0, color='red')

        plt.savefig(pf.target_path + 'ids.png')
        tf.plt.show()
    fig2, ax1 = tf.plt.subplots(dpi=150)
    ax1.plot(mapped_pnts[:, 0, current_frame], mapped_pnts[:, 1, current_frame], 'k.', markersize=6)
    ax1.set_aspect('equal')

    for m in range(0, mapped_pnts.shape[0]):
        ax1.annotate(str(m), (mapped_pnts[m, 0, current_frame] + 0.05, mapped_pnts[m, 1, current_frame] - 0.1),
                     fontsize=6.0, color='red')
        # Setup for the text box
    axbox = tf.plt.axes([0.1, 0.01, 0.1, 0.05])  # position of the text box
    text_box = tf.TextBox(axbox, 'Point ID:')

    # What to do when text is submitted
    current_pnt_id = [150]

    def submit(text):
        current_pnt_id[0] = int(text)
        tf.plt.close()

    text_box.on_submit(submit)
    tf.plt.show()

    return current_pnt_id[0]


def define_insp_and_exp_frame(pnt, mapped_pnts):
    y = np.squeeze(np.linalg.norm(mapped_pnts[pnt, :, :], axis=0))
    x = np.linspace(0, len(y) - 1, len(y))

    from scipy.signal import find_peaks

    peak_threshold = 0.075
    detrend_filter_halfwidth = 200
    curve_smoothing = 5
    valley_smoothing = 5
    adaptive_factor = 1e8

    # x = np.linspace(0,single_marker_pos.shape[0]-1,single_marker_pos.shape[0])
    # y = np.linalg.norm(single_marker_pos[:],2,axis=1)
    yf = tf.movAverage(y, detrend_filter_halfwidth)
    yaf = tf.adaptiveMovAverage(y, detrend_filter_halfwidth, adaptive_factor)

    ax2 = tf.plt.subplot(211)
    ax2.plot(x, y, '-', label='original trace')
    ax2.plot(x, yf, 'r-', label='standard moving average')
    ax2.plot(x, yaf, 'g-', label='adaptive moving average')
    ax2.set_title('background correction trace ' + str(pnt))
    ax2.legend()

    corr_values = tf.movAverage(y - yaf, curve_smoothing)
    peaks, _ = find_peaks(corr_values, height=peak_threshold, distance=22)
    interv = tf.find_intervals(tf.movAverage(corr_values, valley_smoothing), peaks, peak_threshold)
    print('peaks =', peaks)
    print('intervals =', interv)
    ax2 = tf.plt.subplot(212)
    ax2.plot(x, corr_values, '-', label='corrected trace')
    ax2.plot(peaks, corr_values[peaks], 'rx', markersize=12, label='detected peaks')
    ax2.plot(interv, corr_values[interv], 'b*', markersize=12, label='detected intervals')
    ax2.plot((x[0], x[len(x) - 1]), (peak_threshold, peak_threshold), 'r--', label='peak detection lvl')
    ax2.legend()
    ax2.set_title('corrected trace')

    # df = pd.DataFrame({'X':x,'Y':y,'Yf':yf})
    # df.to_csv(export_path+'/'+mouse_name+'_curves_pnt_{}.csv'.format(current_pnt),columns=None)

    # Setup for the text boxes
    axbox1 = tf.plt.axes([0.1, 0.01, 0.1, 0.05])  # position of the first text box
    text_box1 = tf.TextBox(axbox1, 'ref_insp')

    axbox2 = tf.plt.axes([0.3, 0.01, 0.1, 0.05])  # position of the second text box
    text_box2 = tf.TextBox(axbox2, 'ref_exp')

    # Stores values from text boxes
    text_values = {}

    # Callbacks for text boxes
    def submit1(text):
        text_values['ref_insp'] = int(text)
        check_submission()

    def submit2(text):
        text_values['ref_exp'] = int(text)
        check_submission()

    # Checks if both text boxes have values before closing
    def check_submission():
        if 'ref_insp' in text_values and 'ref_exp' in text_values:
            tf.plt.close()

    text_box1.on_submit(submit1)
    text_box2.on_submit(submit2)
    tf.plt.show()

    return text_values.get('ref_insp'), text_values.get('ref_exp')


def interpolate_vertex(mapped_pnts, index_map, r, c, grid_size_cols):
    x= 0
    return

def get_neighbours(r, c):
        return [
            int((r - 1) * grid_size_cols + c),
            int(r * grid_size_cols + (c - 1)),
            int((r + 1) * grid_size_cols + c),
            int(r * grid_size_cols + (c + 1))
        ]

def repair_holes(mapped_pnts, index_map, grid_size_cols):
    #look up table for the IDs
    id_to_index = {ID: index for index, ID in enumerate(index_map)}
    # Sort index_map and mapped_pnts based on index_map
    sorted_indices = sorted(range(len(index_map)), key=lambda k: index_map[k])
    index_map_sorted = copy.deepcopy([index_map[i] for i in sorted_indices])
    mapped_pnts_int = copy.deepcopy(mapped_pnts)
    index_map_int = copy.deepcopy(index_map)
    gaps = []

    for i in range(1, len( index_map_sorted)):
        if  index_map_sorted[i] -  index_map_sorted[i - 1] > 1:
            # There's a gap between consecutive IDs in index_map
            gaps.append(index_map_sorted[i - 1] + 1)

    for ID in gaps:
        r = np.floor(ID / grid_size_cols)
        c = ID % grid_size_cols

        neighbours = get_neighbours(int(r), int(c))

        # If all neighbors of this gap ID are present, interpolate
        if all([neigh in id_to_index for neigh in neighbours]):
            interpolated_values = np.zeros((3, mapped_pnts_int.shape[2]))
            #xn1 = mapped_pnts_int[id_to_index[3242], 0, 0]
            #yn1 = mapped_pnts_int[id_to_index[3242], 1, 0]
            #zn1 = mapped_pnts_int[id_to_index[3242], 2, 0]

            for frame in range(mapped_pnts_int.shape[2]):
                x_sum = sum([mapped_pnts_int[id_to_index[neigh], 0, frame] for neigh in neighbours])
                y_sum = sum([mapped_pnts_int[id_to_index[neigh], 1, frame] for neigh in neighbours])
                z_sum = sum([mapped_pnts_int[id_to_index[neigh], 2, frame] for neigh in neighbours])

                interpolated_values[:, frame] = np.array([x_sum, y_sum, z_sum]) / 4

            mapped_pnts_int = np.vstack([mapped_pnts_int, interpolated_values.reshape(1, 3, -1)])
            index_map_int.append(ID)
            id_to_index = {ID: index for index, ID in enumerate(index_map_int)}  # Update mapping

    return mapped_pnts_int, index_map_int


# generate faces
def polygonize(mapped_pnts, index_map):
    vertex = []
    face = []
    vertex_id_map = []

    for ID in index_map:
        r = np.floor(ID / grid_size_cols)
        c = ID % grid_size_cols

        # For upper right triangle
        if (c < (grid_size_cols - 1)) and \
                ((r + 1) * grid_size_cols + c in index_map) and \
                (r * grid_size_cols + c + 1 in index_map):
            p1_id, vertex, vertex_id_map = find_or_add(vertex, vertex_id_map, r, c, index_map, mapped_pnts)
            p2_id, vertex, vertex_id_map = find_or_add(vertex, vertex_id_map, r + 1, c, index_map, mapped_pnts)
            p3_id, vertex, vertex_id_map = find_or_add(vertex, vertex_id_map, r, c + 1, index_map, mapped_pnts)
            face.extend([3, p1_id, p2_id, p3_id])

        # For lower left triangle
        if (c > 0) and \
                ((r - 1) * grid_size_cols + c in index_map) and \
                (r * grid_size_cols + c - 1 in index_map):
            p1_id, vertex, vertex_id_map = find_or_add(vertex, vertex_id_map, r, c, index_map, mapped_pnts)
            p2_id, vertex, vertex_id_map = find_or_add(vertex, vertex_id_map, r - 1, c, index_map, mapped_pnts)
            p3_id, vertex, vertex_id_map = find_or_add(vertex, vertex_id_map, r, c - 1, index_map, mapped_pnts)
            face.extend([3, p1_id, p2_id, p3_id])

    return vertex, face


def compute_distances(reference_mesh, other_mesh):
    # Compute the centers of each cell in the other_mesh
    centers = other_mesh.cell_centers().points

    # For each center, find its closest point on the reference_mesh and compute the distance
    distances = []
    for center in centers:
        closest_point = reference_mesh.find_closest_point(center)
        distance = center[2] - reference_mesh.points[closest_point][2]
        distances.append(distance)

    return np.array(distances)


def visualize_difference(ref_mesh, motion_mesh, distances):
    # Compute distances
    distances = compute_distances(ref_mesh, motion_mesh)
    motion_mesh['Distances'] = distances
    cmap = 'RdBu_r'
    # Visualize
    plotter = pv.Plotter()
    plotter.add_mesh(motion_mesh, scalars='Distances', cmap=cmap, clim=[-np.max(distances), np.max(distances)])
    plotter.show()
    return motion_mesh

def filter_bad_faces(m, area_th, angle_th):
    #created a mask for face angles

    # get triangles
    v0 = m.points[m.faces.reshape(-1, 4)[:, 1]]
    v1 = m.points[m.faces.reshape(-1, 4)[:, 2]]
    v2 = m.points[m.faces.reshape(-1, 4)[:, 3]]
    # get triangle side vectors
    a = v1-v2
    b = v0-v2
    c = v1 - v0
    d = v2 - v0
    #get triangle angles
    angles = []
    for aa, bb, cc, dd in zip(a, b, c, d):
        alpha = np.arccos(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))
        beta = np.arccos(np.dot(cc, dd) / (np.linalg.norm(cc) * np.linalg.norm(dd)))
        gamma = np.pi - alpha - beta
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)
        gamma = np.rad2deg(gamma)
        if (alpha<angle_th or beta<angle_th or gamma<angle_th):
            angles.append(True)
        else: angles.append(False)

    # create a mask for face areas
    areas = m.compute_cell_sizes()["Area"]

    #make a mask
    max_ar_sor = np.sort(areas)
    areas = areas > max_ar_sor[int(area_th/100*len(max_ar_sor))]
    #combine mask
    mask = np.logical_and(areas,angles)
    #cleanup
    faces = m.faces.reshape((-1, 4))

    # Filter out the unwanted faces based on the delete_mask
    filtered_faces = faces[~mask]

    # Constructing a new PolyData with the same points and the filtered faces
    clean_mesh = pv.PolyData(m.points, filtered_faces, deep=True)

    return clean_mesh

def polygonize_frame(frame, mapped_pnts, L, verbose=False, export_name=None, mouse=''):
    print(mapped_pnts[0,:,frame])
    vertex_l, face_l = polygonize(mapped_pnts[:, :, frame], L)
    mesh = pv.PolyData(vertex_l, face_l, int(len(face_l) / 3))
    filtered_mesh  = filter_bad_faces(mesh, 80, 15)
    # print(vertex_l)
    if not (export_name is None):
        mesh.save(export_name)
        #filtered_mesh.save(pf.target_path+ '/' +str(mouse) + '_filtered.ply')
    # current_frame = frame
    if verbose:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh)
        # Add scale bars (axes)
        plotter.add_axes(at_origin=True, show_edges=True)
        # Set title
        plotter.set_title('mouse:' + mouse + ' frame:' + str(frame))
        plotter.show()
    return filtered_mesh
