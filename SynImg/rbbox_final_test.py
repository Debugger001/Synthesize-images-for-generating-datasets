import cv2
import numpy as np
import math

image_dir = "/Users/pro/Desktop/0_box.png"

ins_mask = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
colorimg = cv2.imread("/Users/pro/Desktop/0box.jpg")

# gray_value = 16

# pixels = []

# mask_gray = np.zeros(ins_mask.shape)
# mask_gray.fill(gray_value)
# tf = (mask_gray == ins_mask)
# print(tf)
# pixels = ins_mask * tf
# print(pixels)
coord = []
for i in range(512):
    temp = []
    for j in range(512):
        temp.append([i,j])
    temp = np.array(temp)
    coord.append(temp)
coord = np.array(coord)

# for i in range(len(ins_mask)):
#     for j in range(len(ins_mask)):
#         if ins_mask[i][j] == gray_value:
#             pixels.append([i,j])
#             colorimg[i][j] = [0,0,255]

rbboxes = []
gray_value = 1
isexists = True

while isexists:
    pixels = []
    isexists = False
    # for i in range(len(ins_mask)):
    #     temp = []
    #     for j in range(len(ins_mask)):
    #         temp.append([i,j])
    #         if ins_mask[i][j] == gray_value:
    #             pixels.append([i,j])
    #             isexists = True
    thisgray = np.zeros((512,512))
    thisgray.fill(gray_value)
    pixels = coord[(thisgray == ins_mask)]
    if len(pixels) == 0:
        isexists = False
    else:
        isexists = True

    if isexists:
        rbbox = cv2.minAreaRect(np.array(pixels))
        x,y = rbbox[0]
        w,h = rbbox[1]
        angle = rbbox[2] * math.pi / 180
        angle = -angle + math.pi / 2
        # class_id = sem_mask[pixels[0][0]][pixels[0][1]]
        gray_value += 1
        rbboxes.append([x,y,w,h,angle])

# colorins_mask[int(x)][int(y)] = [0,255,0]

# rbbox = ((x,y),(w,h),angle)

# box = cv2.boxPoints(rbbox)
# box = np.int0(box)

# ur = [x+h/2*math.cos(angle)+w/2*math.sin(angle), y-h/2*math.sin(angle)+w/2*math.cos(angle)]
# dr = [x+h/2*math.cos(angle)-w/2*math.sin(angle), y-h/2*math.sin(angle)-w/2*math.cos(angle)]
# ul = [x-h/2*math.cos(angle)+w/2*math.sin(angle), y+h/2*math.sin(angle)+w/2*math.cos(angle)]
# dl = [x-h/2*math.cos(angle)-w/2*math.sin(angle), y+h/2*math.sin(angle)-w/2*math.cos(angle)]
for rbbox in rbboxes:

    x,y,w,h,angle = rbbox

    ur = [y-h/2*math.sin(angle)+w/2*math.cos(angle), x+h/2*math.cos(angle)+w/2*math.sin(angle)]
    dr = [y-h/2*math.sin(angle)-w/2*math.cos(angle), x+h/2*math.cos(angle)-w/2*math.sin(angle)]
    ul = [y+h/2*math.sin(angle)+w/2*math.cos(angle), x-h/2*math.cos(angle)+w/2*math.sin(angle)]
    dl = [y+h/2*math.sin(angle)-w/2*math.cos(angle), x-h/2*math.cos(angle)-w/2*math.sin(angle)]

    box = [dl,ul,ur,dr]
    box = np.int0(box)
    # print(box)
    cv2.drawContours(colorimg, [box], 0, (0,0,255), 2)

# x, y, w, h = bbox
# cv2.rectangle(colorins_mask, (y, x), (y+h, x+w), (0, 255, 0), 2)

cv2.imwrite('contours.png', colorimg)
