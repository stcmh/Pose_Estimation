import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'checkerVedio.mp4'
K = np.array([[627.14888199, 0, 213.31259762],
              [0, 625.21197361, 367.41067077],
              [0, 0, 1]])
dist_coeff = np.array([-0.0715358, 0.04850487,  0.00205158, -0.0074781, 0.12175841])
board_pattern = (8, 6)
board_cellsize = 0.03
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

def create_AR(center=(4.5, 3.5), height=2.0, radius=0.5, segments=16):
    
    cx, cy = center
    h = height
    pts = []

    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        pts.append([x, y, 0])
        pts.append([x, y, -h * 0.6])


    pts.append([cx, cy, -h * 0.6 - radius])  
    for i in range(segments):
        theta = 2 * np.pi * i / segments
        x = cx + radius * 0.6 * np.cos(theta)
        y = cy + radius * 0.6 * np.sin(theta)
        z = -h * 0.6 - radius * 0.3
        pts.append([x, y, z])

    return board_cellsize * np.array(pts, dtype=np.float32)

AR_points = create_AR()

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the box on the image
        proj_AR, _ = cv.projectPoints(AR_points, rvec, tvec, K, dist_coeff)
        proj_AR = np.int32(proj_AR).reshape(-1, 2)

        for i in range(0, 32, 2): 
            cv.line(img, proj_AR[i], proj_AR[i + 1], (255, 200, 0), 2)

        cv.circle(img, proj_AR[32], 5, (0, 100, 255), -1) 
        for i in range(33, len(proj_AR)):
            cv.circle(img, proj_AR[i], 2, (100, 100, 255), -1)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()
