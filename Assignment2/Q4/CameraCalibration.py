import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = [] 

Calibration_Corners_Img = []

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

images = glob.glob('./images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret == True:
        objpoints.append(objp)
       
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
        Calibration_Corners_Img.append(img)
    
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

fx = mtx[0, 0]
fy = mtx[1, 1]

cx = mtx[0, 2]
cy = mtx[1, 2]

skew = mtx[0, 1]

print("Intrinsic Camera Parameters: \n")
print("Focal Length (fx, fy):", fx, ",", fy)
print("Principal Point (cx, cy):", cx, ",", cy)
print("Skew Parameter:", skew)

print("\nExtrinsic Camera Parameters")

for i in range(len(rvecs)):
    print("Image", i+1, ":")
    print("Translation Vector:")
    print(tvecs[i])

    rmat, _ = cv2.Rodrigues(rvecs[i])

    print("Rotation Matrix:")
    print(rmat)
    
print("\nEstimated Radial Distortion Coefficients:", dist)

for i in range(5):
    img = cv2.imread(images[i])
    undistorted_img = cv2.undistort(img, mtx, dist)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Distorted Image')

    axes[1].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Undistorted Image')

    plt.tight_layout()
    plt.show()

print("Reprojection Error: \n")

errors = []
for i in range(len(objpoints)):
    imgpoints_reproj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints_reproj, cv2.NORM_L2) / len(imgpoints_reproj)
    errors.append(error)

    print("Image", i+1, ":")
    print(error)

mean = np.mean(errors)
std_dev = np.std(errors)

print("Mean Error:", mean)
print("Standard Deviation of Errors:", std_dev)

plt.figure(figsize=(8, 6))
plt.bar(range(len(errors)), errors)
plt.xlabel('Image Index')
plt.ylabel('Reprojection Error')
plt.title('Reprojection Error for Each Image')
plt.show()

for i in range(len(objpoints)):

    imgpoints_reprojected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    
    img = Calibration_Corners_Img[i]
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.plot(imgpoints[i][:, 0, 0], imgpoints[i][:, 0, 1], 'ro', markersize=6) 
    plt.title(f'Image {i+1} - Detected Corners')
    plt.axis('off')
    
    for j in range(len(imgpoints_reprojected)):
        plt.plot(imgpoints_reprojected[j, 0, 0], imgpoints_reprojected[j, 0, 1], 'bx', markersize=6)
    
    plt.show()

checkerboard_normal = []

for i in range(len(rvecs)):

    R = cv2.Rodrigues(rvecs[i])[0]

    normal_checkerboard = np.array([[0], [0], [1]])
    normal_camera = np.dot(R, normal_checkerboard)
    checkerboard_normal.append(normal_camera)

for i, normal in enumerate(checkerboard_normal):
    print(f"Image {i+1}: Normal in camera coordinate frame of reference= {normal.flatten()}")