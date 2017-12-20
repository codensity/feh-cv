import argparse
import numpy as np
import cv2
import json
import PIL
import sys
import tesserocr
import threading

parser = argparse.ArgumentParser(description="OCR for Fire Emblem Heroes")
parser.add_argument("image_filename", metavar="IMAGE", type=str,
    help="Input image to read")
parser.add_argument("-d", "--debug", action="store_true",
    help="Display intermediate images in windows")
args = parser.parse_args()

enable_image_debug_windows = args.debug

def draw_rect_into(img, corners, color=0):
    line_thickness = 2
    line_type = 8
    cv2.rectangle(img, corners[0], corners[1], color, line_thickness, line_type)

def image_size(img):
    s = img.shape # height, width
    return (s[1], s[0]) # width, height

def rect_corners(rect):
    origin, size = rect
    return (origin, (origin[0] + size[0], origin[1] + size[1]))

def corners_rect(corners):
    return (corners[0], (corners[1][0] - corners[0][0], corners[1][1] + corners[0][1]))

def v2_add(a, b):
    return (a[0] + b[0], a[1] + b[1])

widest_window_in_column = 0
last_window_offset = [0, 20]

def display_image_window(name, img):
    global widest_window_in_column, last_window_offset, enable_image_debug_windows
    if not enable_image_debug_windows:
        return
    h = img.shape[0]
    w = img.shape[1]
    h += 22 # title bar
    w = max(w, 200)
    cv2.imshow(name, img)
    cv2.moveWindow(name, last_window_offset[0], last_window_offset[1])
    last_window_offset[1] += h
    if last_window_offset[1] > 1000:
        last_window_offset[1] = 0
        last_window_offset[0] += widest_window_in_column
    widest_window_in_column = max(widest_window_in_column, w)

def match_template(img, templ):
    match_method = cv2.TM_SQDIFF
    res = cv2.matchTemplate(img, templ, match_method)
    res2 = cv2.normalize(res, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC1)
    #display_image_window("matchTemplate", res2)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res2)
    if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    return rect_corners((matchLoc, image_size(templ)))

DEFAULT_READER_THRESHOLD = 150

class TextReader:
    def __init__(self):
        self.stats = {}
        self.threads = []

    def read_rect(self, name, offset, size, threshold=DEFAULT_READER_THRESHOLD):
        #print("reading " + name)
        r = (offset, size)
        cs = rect_corners(r)
        draw_rect_into(display_img, cs, color=white)
        num_img = img[cs[0][1]:cs[1][1], cs[0][0]:cs[1][0]]
        #thresh, bw_num_img = cv2.threshold(num_img[:,:,1], 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresh, bw_num_img = cv2.threshold(num_img[:,:,1], threshold, 255, cv2.THRESH_BINARY_INV)
        display_image_window(name, bw_num_img)
        bw_num_pil_img = PIL.Image.fromarray(bw_num_img)
        def run_ocr():
            value = tesserocr.image_to_text(bw_num_pil_img)
            self.stats[name] = value.rstrip()
            #print("read    " + name)
        thread = threading.Thread(target=run_ocr)
        thread.start()
        self.threads.append(thread)

    def finish(self):
        for t in self.threads:
            t.join()
        return self.stats

def read_rects(rects, reader, default_size):
    for r in rects:
        name = r[0]
        base_rect = r[1]
        offset = r[2]
        if len(r) > 3:
            more = r[3:]
        else:
            more = [default_size]
        reader.read_rect(name, v2_add(base_rect[0], offset), *more)

img_filename = args.image_filename

img = cv2.imread(img_filename)
plus_img = cv2.imread("plus.png")
right_red_img = cv2.imread("right-red.png")
right_blue_img = cv2.imread("right-blue.png")
detail_back = cv2.imread("detail-back.png")
detail_labels = cv2.imread("detail-labels.png")

orig_h, orig_w, orig_channels = img.shape
target_w = 640
target_h = 1136
new_w = target_h * orig_w / orig_h
img = cv2.resize(img, (new_w, target_h))

detail_back_corners = match_template(img, detail_back)

is_detail_screen = detail_back_corners[0][1] < 50

white = (255, 255, 255)
display_img = img.copy()
reader = TextReader()

small_num_rect_size = (40, 27)

skill_keys = ["weapon", "assist", "special", "a", "b", "c", "s"]

if is_detail_screen:
    detail_labels_corners = match_template(img, detail_labels)
    draw_rect_into(display_img, detail_back_corners)
    draw_rect_into(display_img, detail_labels_corners)

    #print("templates matched")

    title_size = (298, 48)
    name_size = (238, 48)
    name_thresh = 180
    stat_size = (98, 32)
    stat_s_thresh = 160

    cs = detail_labels_corners
    rects = [
        ("level", cs, (46, 0), stat_size),
        ("name", cs, (-25, -92), name_size, name_thresh),
        ("title", cs, (-76, -162), title_size, name_thresh),
    ]

    row_height = 44
    for i, name in enumerate(["hp", "atk", "spd", "def", "res", "sp", "hm"]):
        rects.append((name, cs, (48, 55 + i * row_height), stat_size))

    for i, name in enumerate(skill_keys):
        if name == "s":
            thresh = stat_s_thresh
        else:
            thresh = DEFAULT_READER_THRESHOLD
        rects.append((name, cs, (269, 55 + i * row_height), (215, 32), thresh))

    read_rects(rects, reader, default_size=small_num_rect_size)
    stats = reader.finish()

else: # is top info
    plus_corners = match_template(img, plus_img)
    right_red_corners = match_template(img, right_red_img)
    right_blue_corners = match_template(img, right_blue_img)

    #print("templates matched")

    if abs(right_red_corners[1][1] - plus_corners[1][1]) < abs(right_blue_corners[1][1] - plus_corners[1][1]):
        right_corners = right_red_corners
    else:
        right_corners = right_blue_corners

    draw_rect_into(display_img, detail_back_corners)
    draw_rect_into(display_img, plus_corners)
    #draw_rect_into(display_img, right_red_corners, (0, 0, 255))
    #draw_rect_into(display_img, right_blue_corners, (255, 0, 0))
    draw_rect_into(display_img, right_corners)

    hp_current_rect = (v2_add(right_corners[0], (-402, 26)), (53, 35))
    rects = [
        ("level", right_corners, (-188, -25), (56, small_num_rect_size[1])),
        ("hp_current", hp_current_rect, (0, 0), hp_current_rect[1]),
        ("hp_max", hp_current_rect, (79, 2)),
        ("name", hp_current_rect, (-40, -47), (175, 27)),
    ]

    for name, ix, iy in [("atk", 0, 0), ("spd", 1, 0), ("def", 0, 1), ("res", 1, 1)]:
        rects.append((name, hp_current_rect, v2_add((-4, 48), (118 * ix, 32 * iy))))

    row_height = 41
    for i, name in enumerate(["weapon", "assist", "special"]):
        rects.append((name, hp_current_rect, (205, -2 + row_height * i), (190, 27)))

    read_rects(rects, reader, default_size=small_num_rect_size)
    stats = reader.finish()

def print_stats(s):
    print(s["name"] + " []")
    if "hp_current" in s:
        hp = "%s/%s" % (s["hp_current"], s["hp_max"])
    else:
        hp = s["hp"]
    print("%s HP / %s ATK / %s SPD / %s DEF / %s RES"
        % (hp, s["atk"], s["spd"], s["def"], s["res"]))
    for k in skill_keys:
        if len(k) == 1:
            pk = "Passive " + k.title()
        else:
            pk = k.title()
        if k in s and len(s[k]) > 0:
            print("%s: %s" % (pk, s[k]))

print(json.dumps(stats))
print_stats(stats)

if enable_image_debug_windows:
    display_image_window("image", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("")
