import cv2
import tensorflow as tf
import numpy as np
from scipy.linalg import hadamard
import math as m
import os

std_size = 28       # for resizing images
n_corners = 100     # number of corners to detect
model = tf.keras.models.load_model('SSD_letter.keras') # type: ignore
borders = []

## image handling functions
def read_img(img_path): # read image, equalize histogram
    def eq_hist(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        return img
    
    img = cv2.imread(img_path)
    return eq_hist(img)

def resize_img(img): # extend border to square, resize to std_size
    img = cv2.copyMakeBorder(img, 0, max(img.shape) - img.shape[0], 0, max(img.shape) - img.shape[1], cv2.BORDER_CONSTANT, value=[255])
    img = cv2.resize(img, (std_size, std_size))
    return img

def show_img(img, title = 'image'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


''' Legacy code
## preprocessing functions
# Binarization: local average thresholding
# Thinning: Zhang Suen algorithm
# Denoising: median filter, HoughLinesP, inpaint
def preprocess(img):
    def binarize(img, window_size=3, k=0.5):
        # tmp = cv2.ximgproc.niBlackThreshold(img, window_size, k, cv2.THRESH_BINARY, 255)
        img = cv2.bitwise_not(img)
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, window_size, 0)

    def thinning(binary_img):
        # tmp = cv2.ximgproc.thinning(binary_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        binary_img = (binary_img > 0).astype(np.uint8) * 255
        skeleton = np.ones_like(binary_img) * 255
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        def get_neighbors(x, y):
            return [(x + dx, y + dy) for dx, dy in neighbors]
        def step(image):
            marker = np.zeros(image.shape, dtype=np.uint8)
            for x in range(1, image.shape[0] - 1):
                for y in range(1, image.shape[1] - 1):
                    if image[x, y] == 0:
                        neighbors = [image[nx, ny] for nx, ny in get_neighbors(x, y)]
                        A = (neighbors[0] == 0 and neighbors[1] == 255) + (neighbors[1] == 0 and neighbors[2] == 255) + (neighbors[2] == 0 and neighbors[4] == 255) + (neighbors[4] == 0 and neighbors[7] == 255) + (neighbors[7] == 0 and neighbors[6] == 255) + (neighbors[6] == 0 and neighbors[5] == 255) + (neighbors[5] == 0 and neighbors[3] == 255) + (neighbors[3] == 0 and neighbors[0] == 255)
                        B = sum(neighbors) // 255
                        if 2 <= (8 - B) <= 6 and A == 1 and (neighbors[1] or neighbors[4] or neighbors[6] == 255) and (neighbors[4] or neighbors[6] or neighbors[3] == 255):
                            marker[x, y] = 255
            image[marker == 255] = 0
            return image

        while True:
            before = np.copy(binary_img)
            binary_img = step(binary_img)
            binary_img = step(binary_img)
            if np.all(before == binary_img):
                break
        skeleton[binary_img == 255] = 0
        return skeleton

    def denoise(img):
        # cv2.HoughLinesP
        # cv2.inpaint
        return cv2.medianBlur(img, 5)
    
    for i in range(3):
        img[:, :, i] = denoise(img[:, :, i])
        # img[:, :, i] = binarize(img[:, :, i])
        img[:, :, i] = thinning(img[:, :, i])
    # show_img(np.max(img, axis=2), 'preprocessed')
    return img                      # bgr image
    return np.max(img, axis=2)      # grayscale image

## reference: A Survey on Breaking Technique of Text-Based CAPTCHA
## segmentation functions
# Removes corners that are too close to each other
def non_max_suppression(corners, radius=5):
    if len(corners) == 0:
        return []

    suppressed = np.zeros(len(corners), dtype=bool)
    for i, corner in enumerate(corners):
        if suppressed[i]:
            continue
        x, y = corner[1], corner[0]
        radius_squared = radius ** 2
        for j in range(i + 1, len(corners)):
            other_x, other_y = corners[j][1], corners[j][0]
            if (x - other_x) ** 2 + (y - other_y) ** 2 < radius_squared:
                suppressed[j] = True
    return [corner for i, corner in enumerate(corners) if not suppressed[i]]

# Find intersections of straight edges of text contours
def edge_corner_detection(img):
    edges = cv2.Canny(img, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 18, 5)
    intersections = []
    if lines is not None:
        for line in lines:
            x1, y1, _, _ = line[0]
            intersections.append((x1, y1))
    return intersections

# Segmentation function, based on Edge Corner Detection
#
# Citation: Nachar, R. A., Inaty, E., Bonnin, P. J., and Alayli, Y. (2015) 
#       Breaking down Captcha using edge corners and fuzzy logic segmentation/recognition technique. 
#       Security Comm. Networks, 8: 3995–4012. doi: 10.1002/sec.1316.
# Ref: https://onlinelibrary.wiley.com/doi/epdf/10.1002/sec.1316
def segment(img):
    # edge corner detection
    corners = edge_corner_detection(img)
    corners = non_max_suppression(corners, 7)
    n_corners = len(corners)
    corners.sort(key=lambda x: x[0])

    # show detected ECs
    # print(len(corners))
    # dsp = img.copy()
    # for corner in corners:
    #     cv2.circle(dsp, (corner[0], corner[1]), 1, [100], 2)
    # show_img(dsp, 'corners')

    # EC based segmentation
    captcha_result = []
    recogn_result = None
    best_recogn_result = None
    n_slope = 3
    LB = None
    RB = None
    RBopt = [None, 0]               # [border, match_score]
    best_match = [None, None, 0]    # [RB, ccor, match_score]
    if corners is not None:
        # l_loop = int(n_corners / 5)
        while True:
            SetLB = True
            for ccor in corners[0 : np.int32(len(corners) - 1)]:
                if SetLB:
                    LB = RBopt[0] if RBopt[0] is not None else draw_straight_border(ccor, 0, img.shape)
                    SetLB = False
                else:
                    for i in range(n_slope):
                        RB = draw_straight_border(ccor, i, img.shape)
                        _, match_score = recognition(img.copy(), LB, RB)
                        if match_score > best_match[2]:
                            best_match = [RB, ccor, match_score]
            # print(best_match)
            CBS, n_borders = combination_border_set(corners, LB, best_match)
            if n_borders == 0:
                break
            for b in CBS:
                recogn_result, match_score = recognition(img.copy(), LB, b)
                if match_score > RBopt[1]:
                    RBopt = [b, match_score]
                    best_recogn_result = recogn_result
            borders.append(RBopt[0])
            captcha_result.append(best_recogn_result)

            # if corners[-1] is in RBopt, then break
            if len([corner for corner in RBopt[0] if all(np.equal(corners[-1], corner))]) > 0:
                break

            # remove corners in CBS from corners
            prev_len = len(corners)
            corners = [corner for corner in corners if not is_within_border(corner, LB, RBopt[0])]
            len_diff = prev_len - len(corners)
            RBopt[1] = 0
            best_match[2] = 0
            if len_diff == 0:
                break
            # print(prev_len, len(corners))
            # print(len_diff, captcha_result)
    return captcha_result


# Draws a straight border line from ccor at an angle from y-axis
def draw_straight_border(ccor, i_angle, img_shape):
    x, y = ccor
    angle = [0, np.pi / 8, np.pi / -8][i_angle]
    height, width = img_shape
    if angle == 0:
        return [[x, 0], [x, height]]
    x_up = max(0, np.int32(x - m.tan(angle) * y)) if (np.int32(x - m.tan(angle) * y) < width) else width
    x_down = min(width - 1, np.int32(x + m.tan(angle) * (height - y))) if (np.int32(x + m.tan(angle) * (height - y))) > 0 else 0
    return [[x_up, 0], [x_down, height]]

# Check if cor is within left and right borders
def is_within_border(cor, LB, RB):
    if len(LB) == 0 or len(RB) == 0:
        return False
    
    if len(LB) == 1 or cor[1] <= LB[0][1]:
        x_lb = LB[0][0]
    elif cor[1] >= LB[-1][1]:
        x_lb = LB[-1][0]
    else:
        y1_lb, y2_lb = LB[0], LB[-1]
        for i in range(1, len(LB) - 1):
            if cor[1] >= LB[i][1]:
                y1_lb = LB[i]
                y2_lb = LB[i + 1]
        x_lb = 1.0 * (cor[1] - y2_lb[1]) * (y1_lb[0] - y2_lb[0]) / (y1_lb[1] - y2_lb[1]) + y2_lb[0]
    
    if len(RB) == 1 or cor[1] <= RB[0][1]:
        x_rb = RB[0][0]
    elif cor[1] >= RB[-1][1]:
        x_rb = RB[-1][0]
    else:
        y1_rb, y2_rb = RB[0], RB[-1]
        for i in range(1, len(RB) - 1):
            if cor[1] >= RB[i][1]:
                y1_rb = RB[i]
                y2_rb = RB[i + 1]
        x_rb = 1.0 * (cor[1] - y2_rb[1]) * (y1_rb[0] - y2_rb[0]) / (y1_rb[1] - y2_rb[1]) + y2_rb[0]
    return x_lb <= cor[0] <= x_rb

# Check if two corners are within 10 pixels
def is_near(cor1, cor2):
    x1, y1 = cor1
    x2, y2 = cor2
    return (x1 - x2)**2 + (y1 - y2)**2 <= 100

# Recursive function to find all possible borders
def find_combination(ccor, near_cors, combination, border_set):
    combination.append(ccor)
    border_set.append(combination)
    for c in near_cors:
        if len([corner for corner in combination if all(np.equal(c, corner))]) == 0 and c[1] >= ccor[1] and is_near(ccor, c):
            _ = find_combination(c, near_cors, combination.copy(), border_set)
    return len(border_set)

# Find all possible borders based on best match
def combination_border_set(corners, LB, best_match):
    border_set = []
    near_cors = [c for c in corners if (c[1] >= best_match[1][1])][:np.int32(min((n_corners / 5) - 1, len(corners)))]
    n_borders = find_combination(best_match[1], near_cors, [], border_set)
    return border_set, n_borders

## breaking functions
# Main recognition function, based on EMNIST CNN
def recognition(img, LB, RB):
    left_corner = min(LB[:][0])
    right_corner = max(RB[:][0])
    if (right_corner - left_corner < 8):
        return None, 0
    
    for x in range(left_corner, right_corner):
        for y in range(img.shape[0]):
            if not is_within_border([x, y], LB, RB):
                img[y, x] = 0
    trim_img = img[:, left_corner:right_corner]
    contours, _ = cv2.findContours(trim_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        con_x, con_y, con_w, con_h = cv2.boundingRect(contours[0])
        trim_img = trim_img[con_y:con_y + con_h, con_x:con_x + con_w]

    predictions = model.predict(resize_img(trim_img).reshape(1, std_size, std_size, 1)) # type: ignore

    class_result = parse_emnist(np.argmax(predictions))
    confidence = np.max(predictions)
    return class_result, confidence

# Parse Emnist classes to character
def parse_emnist(class_arg):
    if class_arg > 25:
        if class_arg == 26 or class_arg == 27:
            class_arg -= 26
        elif class_arg <= 32:
            class_arg -= 25
        elif class_arg == 33:
            class_arg -= 20
        elif class_arg < 36:
            class_arg -= 18
        elif class_arg == 36:
            class_arg -= 17
        else:
            return chr((class_arg + 11).astype(int))
    return chr((class_arg + 65))
    
'''

# Intersection over Union
def IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1+w1, x2+w2) - x
    h = min(y1+h1, y2+h2) - y
    if w < 0 or h < 0:
        return 0
    i = w * h
    return i / (w1*h1 + w2*h2 - i)

# Merge RoI with IoU > 0.7
def merge_boxes(boxes): # threshold u = 0.7
    merged_boxes = []
    while boxes:
        box1 = boxes.pop(0)
        to_merge = [box1]
        for box2 in boxes[:]:
            if IoU(box1, box2) > 0.7:
                to_merge.append(box2)
                # boxes.remove(box2)
                boxes = [box for box in boxes if not all(np.equal(box, box2))]
        if len(to_merge) > 1:
            x1 = min([b[0] for b in to_merge])
            y1 = min([b[1] for b in to_merge])
            x2 = max([b[0] + b[2] for b in to_merge])
            y2 = max([b[1] + b[3] for b in to_merge])
            merged_boxes.append([x1, y1, x2 - x1, y2 - y1])
        else:
            merged_boxes.append(box1)
    return merged_boxes

# Instance segmentation from mask rcnn
def segment2(img, curr_label):
    def is_bg(t_rect):
        t_img = img[t_rect[1]:t_rect[1]+t_rect[3], t_rect[0]:t_rect[0]+t_rect[2]]
        n_bg = 0
        for row in t_img:
            for pixel in row:
                if all(pixel == (255, 255, 255)):
                    n_bg += 1
        return n_bg > 0.7 * t_img.shape[0] * t_img.shape[1]
    
    borders.clear()
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()                                # image segmentation
    rects = sorted(rects, key=lambda x: (x[0], x[2]))   # consider RoI with max width = half of image height and slightly taller than wide
    rects = [rect for rect in rects if rect[2] > 12 and rect[3] > 15 and rect[2] < 40 and rect[3] < 50 and not is_bg(rect)]
    rects = merge_boxes(rects)                          # merge regions with IoU > 0.7 (same object) and remove background regions
    captcha_result = {}
    tmp_img = img.copy()

    for [x, y, w, h] in rects:
        if is_bg([x, y, w, h]):
            continue                                    # filter regions with background > 0.7, redundancy check
        trim_img = img[y:y+h, x:x+w]
        class_result, confidence = recogn2(trim_img)    # get NN prediction
        if confidence > 0.7:                            # filter regions with confidence > 0.7
            img[y:y+h, x:x+w] = 255                     # remove region from image, redundancy check
            captcha_result[x + w // 2] = class_result   # store prediction with x-coordinate as key
            borders.append([x, y, w, h])

    # gen new letter_data
    # if len(captcha_result) == len(curr_label):
    #     tmp = [key for key, _ in sorted(captcha_result.items())]
    #     for i in range(len(curr_label)):
    #         for [x, y, w, h] in rects:
    #             if tmp[i] == x + w // 2:
    #                 if not os.path.exists(f'./letter2_data/{curr_label[i]}'):
    #                     os.makedirs(f'./letter2_data/{curr_label[i]}')
    #                 cv2.imwrite(f'./letter2_data/{curr_label[i]}/{i}-{curr_label}.png', tmp_img[y:y+h, x:x+w])

    segment_output = [value for _, value in sorted(captcha_result.items())]
    return ''.join(segment_output)

def recogn2(img):
    img = cv2.resize(img, (300, 300))                   # 'ssd_letter.keras' model takes in a (300, 300, 3) region
    predictions = model.predict(np.expand_dims(img, axis=0))
    class_result = parse_ssd(np.argmax(predictions))
    confidence = np.max(predictions)
    return class_result, confidence

# Parse class index to chr, for 'ssd_letter.keras' model
def parse_ssd(class_arg): # 0-9, a-z
    if class_arg < 10:
        return str(class_arg)
    return chr(class_arg + 87)


def validate(): # validate model, slow loop through ./train_data
    total = 0
    correct = 0
    brkpt = 0
    for file in os.listdir('test_data/test'):
        img = read_img(os.path.join('test_data/test', file))
        # remove salt and pepper noise
        img = cv2.medianBlur(img, 3)
        # remove occlusion lines
        img = cv2.dilate(img, np.ones((5, 5), np.uint8))
        img = cv2.erode(img, np.ones((5, 5), np.uint8))
        # show_img(img, 'preprocessed')
        letters = segment2(img.copy(), [ltr for ltr in file.split('-')[0]])
        print('Actual:', file.split('-')[0], '    Predicted:', letters)
        total += 1
        if letters == file.split('-')[0]:
            correct += 1
        print('score:', correct / total)
        # show_img(img, letters)

if __name__ == '__main__':
    # read data
    file = '2v2n13je-0.png'
    img = read_img(os.path.join('test_data/test', file))
    show_img(img, 'original')
    cv2.imwrite('original.png', img)

    # preprocess image
    img = cv2.medianBlur(img, 3)
    img = cv2.dilate(img, np.ones((5, 5), np.uint8))
    img = cv2.erode(img, np.ones((5, 5), np.uint8))
    show_img(img, 'preprocessed')
    cv2.imwrite('preprocessed.png', img)

    # segment image
    letters = segment2(img.copy(), [ltr for ltr in file.split('-')[0]])
    print('Actual:', file.split('-')[0], '    Predicted:', letters)

    for border in borders:
        cv2.rectangle(img, (border[0], border[1]), (border[0] + border[2], border[1] + border[3]), [175], 2)
    show_img(img, letters)
    cv2.imwrite('result.png', img)

    # show images
    # for border in borders:
    #     for b in border:
    #         cv2.circle(img, (b[0], b[1]), 1, [175], 2)
    # show_img(img, 'original')
    
    # validate()

## Trial keras model results:
# seq: val_acc = 0.8302
# vgg: val_acc = 0.67
# vgg2: val_acc = 0.8
# ssd_letter: val_acc = 0.6586, val_loss = 2.0648
# ssd_letter2: val_acc = 0.6750, val_loss = 1.9737