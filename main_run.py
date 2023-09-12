import tool_functions as tf
import paths_and_flags as pf
import cam_calibration_tools as cct
import cam_decoding_tools as cdt
import meshing_rendering as mr
import dill
import pickle
import os


# calibration
if pf.calibrate:
    cmtx, cRot, cTrans, camera_pairs = cct.do_calibration(True)  # change to false if you don't want visual confirmation
    filename = pf.saved_calib_path + 'Cam_calib.pkl'
    with open(filename, 'wb') as f:
        pickle.dump([cmtx, cRot, cTrans, camera_pairs], f)
        print('session saved as ', filename, 'for calibration ', pf.import_folder)
        print('calibration done')
    calib = [cmtx, cRot, cTrans, camera_pairs]

if not pf.calibrate:
    filename = pf.saved_calib_path + 'Cam_calib.pkl'
    f = open(filename, "rb")
    calib = pickle.load(f)
    f.close()
    print('loaded calibration file: ', filename)
    print('Camera matrices: \n', calib[0])
    print('\n Rotation: \n', calib[1])
    print('\n Translation: \n', calib[2])
    print('\n Camera pairs: \n', calib[3])

# decoding and tracking
# tracks_export_folder = '/Volumes/Optolufu_2/Optolufu_II_Fibrosis/fibrosis_run_II/2023_04_19_day14/Optolufu/analysis/tracks/'
# if not os.path.exists(tracks_export_folder):
#    os.makedirs(tracks_export_folder)

# decoding, tracking, 3D space solving
tracked_data = []
mice = tf.getListOfFiles(pf.import_folder, '', False, True)
print(mice)
# Optional: check one stream to make sure the decoding phase is determined correctly
#E,start,end=cdt.analyze_stream(tf.np.sort(tf.getListOfFiles(str(mice[0]),'tif',True,True))[0],True,'Decoding test')
#print(start,end)


if pf.save_tracks:
    tracked_data = cdt.decode_track_solve_all(mice, calib[0], calib[1], calib[2], Verbose=False)
    dill.dump_module(pf.target_path + 'solved1.db')
if not pf.save_tracks:
    dill.load_module(pf.target_path + 'solved1.db')
mr.mesh_and_render(tracked_data)