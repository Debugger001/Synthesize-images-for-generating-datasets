import json
import numpy as np
import cv2
import math

img_dir = "/disk6/coco/val2017/"

# json_file_dir = "/Users/pro/Desktop/instances_val2017.json"
json_file_dir = "/disk6/coco/annotations/instances_val2017.json"

json_file = open(json_file_dir, encoding='utf-8')
j_file = json.load(json_file)

print(j_file.keys())
# print(j_file["annotations"][0]["segmentation"])

counter = 0
for annotation in j_file["annotations"]:
    if annotation["iscrowd"] == 0:
        # print(annotation["segmentation"])

        # print(len(annotation["segmentation"]))
        # print(annotation["segmentation"][0])

        counter += 1
        if counter == 10:
            print(annotation)
            polygon = annotation["segmentation"][0]
            img_id = annotation["image_id"]
            break
# print(j_file["images"][0])

for image in j_file["images"]:
    if image["id"] == img_id:
        img = cv2.imread(img_dir + image["file_name"])
        print(image)


pixels = np.array(polygon).reshape((int(len(polygon)/2), 2))
pixels = pixels.astype(int)
rbbox = cv2.minAreaRect(np.array(pixels))

x,y = rbbox[0]
w,h = rbbox[1]
angle = rbbox[2] * math.pi / 180
angle = -angle + math.pi / 2

ur = [y-h/2*math.sin(angle)+w/2*math.cos(angle), x+h/2*math.cos(angle)+w/2*math.sin(angle)]
dr = [y-h/2*math.sin(angle)-w/2*math.cos(angle), x+h/2*math.cos(angle)-w/2*math.sin(angle)]
ul = [y+h/2*math.sin(angle)+w/2*math.cos(angle), x-h/2*math.cos(angle)+w/2*math.sin(angle)]
dl = [y+h/2*math.sin(angle)-w/2*math.cos(angle), x-h/2*math.cos(angle)-w/2*math.sin(angle)]

box = [dl,ul,ur,dr]
box = np.int0(box)
# print(box)
cv2.drawContours(img, [box], 0, (0,0,255), 2)

cv2.imwrite('contours_coco.png', img)
