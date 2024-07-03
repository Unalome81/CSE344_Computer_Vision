import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def Keypoint_Descriptor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(gray, None)
    return keypoint, descriptor

def Flann_Matching(Descriptor_list, i, j):
    flann = cv2.FlannBasedMatcher()
    matches_flann = flann.knnMatch(Descriptor_list[i], Descriptor_list[j], k=2)
    print("matches flann = ", len(matches_flann))

    good_flann_matches = []

    for m, n in matches_flann:
        if(m.distance < 0.7 * n.distance):
            good_flann_matches.append(m)

    return good_flann_matches

# Folder containing images
folder_path = 'Images'

# Storing the path of all images
image_paths = [os.path.join(folder_path, img_name) for img_name in os.listdir(folder_path)]

# Importing all images and resizing
Images = []
for img_path in image_paths:
    img = cv2.imread(img_path)
    Images.append(img)

Keypoint_list = []
Descriptor_list = []

# Extracting keypoints and descriptors from images
for img in Images:
    keypoint, descriptor = Keypoint_Descriptor(img)
    Keypoint_list.append(keypoint)
    Descriptor_list.append(descriptor)

# Overlaying keypoints on images
Images_Overlaid_Keypoints = []
for img, key in zip(Images, Keypoint_list):
    Images_Overlaid_Keypoints.append(cv2.drawKeypoints(img, key, None))

plt.figure(figsize=(12, 6))

plt.imshow(cv2.cvtColor(Images_Overlaid_Keypoints[0], cv2.COLOR_BGR2RGB))
plt.title('Image 1 with keypoints')
plt.axis('off')

plt.show()

plt.imshow(cv2.cvtColor(Images_Overlaid_Keypoints[1], cv2.COLOR_BGR2RGB))
plt.title('Image 2 with keypoints')
plt.axis('off')

plt.show()

# Brute force matching of image 1 and 2
bf = cv2.BFMatcher()
matches_bf = bf.match(Descriptor_list[0], Descriptor_list[1])
matches_bf = sorted(matches_bf, key=lambda x: x.distance)
matches_bf_plot = sorted(matches_bf, key=lambda x: x.distance)[:50] 

matched_image_bf = cv2.drawMatches(Images[0], Keypoint_list[0], Images[1], Keypoint_list[1], matches_bf_plot, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(cv2.cvtColor(matched_image_bf, cv2.COLOR_BGR2RGB))
plt.title('Matched Features Image 1 and Image 2(BruteForce) top 50')
plt.axis('off')

plt.show()

# Flann Based matching of Image1, Image1 and Homography Matrix Calculation

matches_flann = Flann_Matching(Descriptor_list, 0, 0)

matched_image_flann = cv2.drawMatches(Images[0], Keypoint_list[0], Images[0], Keypoint_list[0], matches_flann, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

source_points = np.float32([ Keypoint_list[0][m.trainIdx].pt for m in matches_flann ]).reshape(-1,1,2)
destination_points = np.float32([ Keypoint_list[0][m.queryIdx].pt for m in matches_flann ]).reshape(-1,1,2)

M1, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC,5.0)

# Flann Based matching of Image1, Image2 and Homography Matrix Calculation

matches_flann = Flann_Matching(Descriptor_list, 0, 1)

matches_flann_plot = sorted(matches_flann, key=lambda x: x.distance)[:50] 

matched_image_flann = cv2.drawMatches(Images[0], Keypoint_list[0], Images[1], Keypoint_list[1], matches_flann_plot, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

source_points = np.float32([ Keypoint_list[1][m.trainIdx].pt for m in matches_flann ]).reshape(-1,1,2)
destination_points = np.float32([ Keypoint_list[0][m.queryIdx].pt for m in matches_flann ]).reshape(-1,1,2)

M2, _ = cv2.findHomography(source_points, destination_points, cv2.RANSAC,5.0)

plt.imshow(cv2.cvtColor(matched_image_flann, cv2.COLOR_BGR2RGB))
plt.title('Matched Features Image 1 and Image 2 (Flann))')
plt.axis('off')

plt.show()

print("M1 = ", M1)
print("M2 = ", M2)

#Perspective Transformation

warped_img1 = cv2.warpPerspective(Images[0], M1, (Images[1].shape[1], Images[0].shape[0]))
warped_img2 = cv2.warpPerspective(Images[1], M2, (Images[1].shape[1], Images[1].shape[0]))

Side_by_Side = np.hstack((warped_img1, warped_img2))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
plt.title('First Warped Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(warped_img2, cv2.COLOR_BGR2RGB))
plt.title('Second Warped Image')
plt.axis('off')
plt.show()

# Building Panorma
warped_img2 = cv2.warpPerspective(Images[1], M2, (Images[0].shape[1] + Images[1].shape[1], Images[0].shape[0]))

Panorma_1 = np.hstack((warped_img1, warped_img2))

plt.imshow(cv2.cvtColor(Panorma_1, cv2.COLOR_BGR2RGB))
plt.title('Panorma without cropping and blending')
plt.axis('off')

plt.show()

Panorma_2 = cv2.warpPerspective(Images[1], M2, (Images[0].shape[1] + Images[1].shape[1], Images[0].shape[0]))
Panorma_2[0 : Images[1].shape[0], 0 : Images[1].shape[1]] = Images[0]

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(Panorma_2, cv2.COLOR_BGR2RGB))
plt.title('Panorma with cropping and blending')
plt.axis('off')
plt.show()

# Multistitching of images

stitcher = cv2.Stitcher_create()

status, Panorma_3 = stitcher.stitch(Images)

if status == cv2.Stitcher_OK:
  
    plt.imshow(cv2.cvtColor(Panorma_3, cv2.COLOR_BGR2RGB))
    plt.title('Panorma of all images')
    plt.axis('off')
    plt.show()
else:
    print("Stitching failed!")