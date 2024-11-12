import numpy as np
import cv2

import time
import pygame

# initialize pygame
pygame.init()
pygame.mixer.init()

# Sound files
open_sound_path = './sounds/abriu.mp3'
click_sound_path = './sounds/click.mp3'
right_sound_path = './sounds/right_angle.mp3'

# Image files
lock_image_path = './images/combinacao.png'
seven_image_path = './images/seven.png'

def calculate_angle(corner_1, corner_2):

    dx = corner_2[0] - corner_1[0]
    dy = corner_2[1] - corner_1[1]

    # calculate the angle between the two points
    angle = np.degrees(np.arctan2(dy, dx))

    # convert the angle to the range 0-360
    if angle < 0:
        angle += 360

    return angle

def verify_direction(direction, sequence, state, click):

    if abs(direction) > 180:
        click.play()
        direction = -direction/abs(direction)
        if direction != sequence[state]:
            state = 0
            print('errou')

            return state, "errou"
        
        return direction, None

    elif abs(direction) > 2:
        click.play()
        direction = direction/abs(direction)
        if direction != sequence[state]:
            state = 0
            print('errou')
            
            return state, "errou"

        return direction, None

    return direction, None

def verify_right_angle(angle, angle_list, state, opened, right_angle):

    if (angle > angle_list[state] - 2) and (angle < angle_list[state] + 2):
        if state == 3:
            opened.play()
            print('acertou')
            time.sleep(1)
            state += 1
            return state, "abriu"
        else:
            right_angle.play()
            state += 1
            print(angle)
            return state, "right_angle"

    return state, None


def main():

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # variables for the game
    state = 0
    angle_list = [90, 180, 270, 0, 0]
    sequence = [1, -1, 1, -1]
    current_previous_angle = (0, 0)

    # initialize the sound effects
    click = pygame.mixer.Sound(click_sound_path)
    opened = pygame.mixer.Sound(open_sound_path)
    right_angle = pygame.mixer.Sound(right_sound_path)

    # Load the images
    image_lock = cv2.imread(lock_image_path)
    image_seven = cv2.imread(seven_image_path)

    image_seven = cv2.resize(image_seven, (900, 900))

    # Get the limits of the image that will be inserted in the original one
    [l,c,ch] = np.shape(image_lock)
    pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])


    main_loop = True
    image_seven_true = 0

    while True:
        # grap the frame from the threaded video stream and resize it
        ret, frame = cap.read()

        # verify that the frame is not None
        if frame is None:
            print("Frame is None")
            break

        frame = cv2.resize(frame, (900, 900))
        f = frame.copy()
        # Detect the aruco markers in the frame
        markerCorners, markerIds, rejectedImgPoints = arucoDetector.detectMarkers(frame)

        image_corners = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # verify if the markers are detected
        if markerCorners != ():
            for markerCrns, idx in zip(markerCorners, markerIds):
                if idx == 8:

                    # get the corners of the marker
                    corner_1 = markerCrns[0][0]
                    corner_2 = markerCrns[0][1]

                    angle = calculate_angle(corner_1, corner_2)

                    current_previous_angle = (angle, current_previous_angle[0])
                    direction = current_previous_angle[0] - current_previous_angle[1]

                    info, error = verify_direction(direction, sequence, state, click)

                    if error != "errou":
                        direction = info
                    else:
                        state = info

                    state, message = verify_right_angle(angle, angle_list, state, opened, right_angle)  

                    # Destiny points are the corners of the marker
                    pts_dst = markerCrns[0].reshape(-1, 2)

                    # Calculate the homography
                    h, status = cv2.findHomography(pts_src, pts_dst)

                    # Warp source image to destination based on homography
                    warped_image = cv2.warpPerspective(image_lock, h, (frame.shape[1], frame.shape[0]))

                    # I want to put logo on top-left corner, So I create a ROI
                    rows, cols, channels = warped_image.shape
                    roi = frame[0:rows, 0:cols]

                    # Now create a mask of logo and create its inverse mask also
                    img2gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # Now black-out the area of logo in ROI
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    # Take only region of logo from logo image.
                    img2_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)

                    # Put logo in ROI and modify the main image
                    dst = cv2.add(img1_bg, img2_fg)
                    frame[0:rows, 0:cols] = dst

                    if message == "abriu":
                        print("abriu")
                        image_seven_true = 1
                        break
        
                    elif message == "right_angle":
                        print("right_angle")

        if image_seven_true == 1:

            frame = image_seven
            main_loop = False

            
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or main_loop == False:

            if main_loop == False:
                time.sleep(5)
                break
            
            break


if __name__ == "__main__":
    main()


                    
