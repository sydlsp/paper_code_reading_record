import os
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import cv2 as cv
from PIL import Image, ImageDraw

img_path="00006.jpg"

"""
{"image_id": 6, "category_id": 1, "bbox": [191.9998, 96.9998, 45.0, 57.0], "score": 0.31}
{"image_id": 6, "category_id": 5, "bbox": [233.0003, 0.0, 28.0, 76.9997], "score": 0.41}
{"image_id": 6, "category_id": 1, "bbox": [273.0, 131.0002, 43.0, 58.0], "score": 0.83}


{"image_id": 6, "category_id": 1, "bbox": [153.5999, 121.2497, 56.25, 45.6], "score": 0.31}
{"image_id": 6, "category_id": 5, "bbox": [186.4003, 0.0, 35.0, 61.5997], "score": 0.41}
{"image_id": 6, "category_id": 1, "bbox": [218.4, 163.7503, 53.75, 46.4], "score": 0.83}
"""
img = plt.imread(img_path)
# plt.imshow(img)
# plt.gca().add_patch(pat.Rectangle((205, 72), 170, 147, linewidth=2, edgecolor='r', facecolor='None'))
fig, ax = plt.subplots(1)
ax.imshow(img)


# Rectangle 坐标的参数格式为左上角（x, y），width, height。
rec = pat.Rectangle((273.0,131), 58, 43, linewidth=2, edgecolor='b', facecolor='None')
rec1=pat.Rectangle((233,0), 77.0, 28.0, linewidth=2, edgecolor='b', facecolor='None')
ax.add_patch(rec)
ax.add_patch(rec1)
plt.imshow(img)
plt.show()
fig.savefig('new_test_6.jpg')