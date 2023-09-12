import numpy as np
from skimage import draw, io
import PIL
#in px, dictated by the used projector
#pattern_w = 912
#pattern_h = 1140

pattern_w =  3840
pattern_h = 2160
#note, projector stretches in w dimension by factor of 2
element_d = 5 # target diameter
edge_edge = element_d*2 # target edge to edge distance
margin = 100
r= element_d/2
cc_dist = edge_edge + 2*r
e_num_h = int(np.ceil((pattern_h-2*margin)/cc_dist))
#e_num_w = int(np.ceil((pattern_w-margin)/(cc_dist/2))) # mouse
e_num_w = int(np.ceil((pattern_w-2*margin)/cc_dist)) # human
element_number =  e_num_w * e_num_h
num_pattern = int(np.log2(element_number)+1)
print (e_num_h)
print (e_num_w)
print (element_number)

canvas = np.zeros((pattern_h, pattern_w,num_pattern+1), dtype = bool)

for c in range (e_num_w):
    for r in range (e_num_h):
        #cx = c * cc_dist/2 + margin/2 # mouse
        cx = c * cc_dist + margin # human
        cy = r * cc_dist + margin
        #w = element_d / 2 # for mouse projector to compensate the stretch
        w = element_d # for regular projector
        h = element_d
        rr, cc = draw.ellipse(cy, cx, h/2, w/2, rotation=np.deg2rad(0))
        canvas[rr, cc, 0] = 1
img = PIL.Image.fromarray(canvas[:,:,0])
img.save(str('/Users/angelika.svetlove/research files/Optolufu_asthma/encoding_planes_patient/pattern_pil{:02d}.bmp').format(0), bits = 1, optimize = True)
for p in range(num_pattern):
    marker = 0
    for c in range(e_num_w):
        for r in range(e_num_h):
            if (marker>>p)%2 == 1:
                #cx = c * cc_dist / 2 + margin / 2 #mouse
                cx = c * cc_dist + margin  # human
                cy = r * cc_dist + margin
                #w = element_d / 2
                w = element_d  # for regular projector
                h = element_d
                rr, cc = draw.ellipse(cy, cx, h / 2, w / 2, rotation=np.deg2rad(0))
                canvas[rr, cc, p+1] = 1
            marker = marker+1
    img = PIL.Image.fromarray(canvas[:, :, p+1])
    img.save(str('/Users/angelika.svetlove/research files/Optolufu_asthma/encoding_planes_patient/pattern_pil{:02d}.bmp').format(p+1), bits=1, optimize=True)

marker = 0
export_24bit = np.zeros((pattern_h,pattern_w,3), dtype='uint8')
for c in range(e_num_w):
    for r in range(e_num_h):
        #cx = c * cc_dist / 2 + margin / 2 # mouse
        cx = c * cc_dist + margin  # human
        cy = r * cc_dist + margin
        w = element_d / 2
        w = element_d  # for regular projector
        h = element_d
        rr, cc = draw.ellipse(cy, cx, h / 2, w / 2, rotation=np.deg2rad(0))
        export_24bit[rr, cc, 0] = (marker*2+1) & 0x0000FF
        export_24bit[rr, cc, 1] = ((marker*2+1) & 0x00FF00) >> 8
        export_24bit[rr, cc, 2] = ((marker*2+1) & 0xFF0000) >> 16
        marker = marker+1
img = PIL.Image.fromarray(export_24bit, mode="RGB")
img.save(str('/Users/angelika.svetlove/research files/Optolufu_asthma/encoding_planes_patient/pattern_24pil{:02d}.bmp').format(0))






