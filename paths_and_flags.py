#################################################################################
# parameters
# where the camera recordings are stored
import_folder = "/Users/angelika.svetlove/research files/Optolufu_fibrosis/test/"
# where everything should be saved
target_path = "/Users/angelika.svetlove/research files/Optolufu_fibrosis/out/"
# folder where all the calibration images are stored
import_calib_folder = '/Users/angelika.svetlove/research files/Optolufu_fibrosis/test/rot_inv_pattern copy/'
# path to save or to pull from
saved_calib_path = "/Users/angelika.svetlove/research files/Optolufu_fibrosis/out/"

calibrate = False # set to True if you want to calibrate again, otherwise loads calibration in saved_calib_path
save_tracks = False # it True then pickles the decoded, tracked and solved session. If False, loads one from target_path.

use_KLT = True # True if using Kaerhun-Loewe Transform. Will reorient the mesh in the direction of most displacement.
# If False, mesh will be displayed in the space of the reference camera.
generate_animation = True
amplified = True
#################################################################################