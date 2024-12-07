import numpy as np
import csv
import cv2
import pyautogui
from pynput.mouse import Button, Controller
import mediapipe as mp
from keras.api.models import load_model

import threading


# Screen position
screenWidth, screenHeight = pyautogui.size()

def get_screen_position(lm_x, lm_y, detect_margin):
    return max(0.001, min(1 - (lm_x-detect_margin) / (1-2*detect_margin), 1-0.001)) * screenWidth, max(0.001, min((lm_y-detect_margin) / (1-2*detect_margin), 1-0.001)) * screenHeight


# Smoothing parameters
alpha = 0.4
prev_x, prev_y = pyautogui.position()
smoothed_x, smoothed_y = 0, 0

def mouseMove(mouse, x, y):
    global prev_x, prev_y, smoothed_x, smoothed_y
    smoothed_x = alpha * x + (1 - alpha) * prev_x
    smoothed_y = alpha * y + (1 - alpha) * prev_y
    prev_x, prev_y = smoothed_x, smoothed_y
    mouse.position = (smoothed_x, smoothed_y)

def mouseDown(mouse):
    mouse.press(Button.left)
    print("Mouse Down")

def mouseUp(mouse):
    mouse.release(Button.left)
    print("Mouse Up")

def mouseClick(mouse):
    mouse.press(Button.left)
    mouse.release(Button.left)
    print("Mouse Click")

def normalizeLandmarks(landmarks):
    landmarks = np.array(landmarks)
    center = landmarks[0]
    normalized_landmarks = landmarks - center
    max_dist = np.max(np.linalg.norm(normalized_landmarks[:, None] - normalized_landmarks, axis=2))
    # max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    normalized_landmarks /= max_dist

    return normalized_landmarks.flatten()


def logKeypoint(label, landmarks):
    file_path = "added_keypoint.csv"
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([label, *landmarks])


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    model = load_model('test_model.keras')

    mouse = Controller()

    detect_margin = 0.2

    cap = cv2.VideoCapture(0)

    pre_class = -1
    lm_x, lm_y = 0, 0
    log_mode = False

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        predicted_class = -1
        
        key = cv2.waitKey(1) & 0xFF

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = np.asarray([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            input_landmarks = normalizeLandmarks(landmarks)
            
            if log_mode:
                if ord('4') <= key <= ord('9'):
                    logKeypoint(int(chr(key)), input_landmarks)
            else:
                input_landmarks = input_landmarks.reshape(1, -1)
                predicted_class = np.argmax(model.predict(input_landmarks, verbose=0))
                if predicted_class == 1:
                    if pre_class == 2:
                        threading.Thread(target=mouseClick, args=(mouse,)).start()
                lm_x, lm_y = get_screen_position(*np.mean(landmarks[[0, 5, 9, 13, 17],:], axis=0), detect_margin)

                if predicted_class == 0:
                    threading.Thread(target=mouseMove, args=(mouse, lm_x, lm_y)).start()
                else:
                    pass

                pre_class = predicted_class
        else:
            pre_class = -1
        

        image = cv2.flip(image, 1)
        image = cv2.resize(image, (800, 600))

        if log_mode:
            cv2.putText(image, "LOG MODE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(image, str(predicted_class), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
        
        cv2.imshow('MediaPipe Hands', image)
        if key == 27:
            break
        elif key == ord('l'):
            log_mode = not log_mode
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()