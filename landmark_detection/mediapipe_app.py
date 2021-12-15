import cv2
import glob
import json
import mediapipe as mp
from kafka import KafkaProducer


prod = KafkaProducer(bootstrap_servers=['localhost:9092'])

# Use this function to send landmarks of a video to Gesture recognition subsystem
def send_landmarks(path = '../../e2e_test_videos/video_1.mp4'):
    return_list = get_landmarks(path)
    json_str = json.dumps(return_list)
    ack = prod.send('ld2gr_topic', bytes(json_str, 'utf-8'))
    print(ack.get())

def format_landmarks(mp_hands, results, image_width, image_height):
    frame_map = {}
    x_list = []
    y_list = []
    for hand_landmarks in results.multi_hand_landmarks:
        hand_landmarks.landmark.pop(mp_hands.HandLandmark.INDEX_FINGER_TIP)
        for k in hand_landmarks.landmark:
            x_list.append(k.x * image_width)
            y_list.append(k.y * image_height)

    frame_map["x"] = x_list
    frame_map["y"] = y_list
    return frame_map

def get_landmarks(path):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # result_map = {}
    return_list = []

    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        vidcap = cv2.VideoCapture(path)
        if vidcap.isOpened():
            success, image = vidcap.read()
            while success:
                # Read an image, flip it around y-axis for correct handedness output (see
                # above).
                image = image
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # Print handedness and draw hand landmarks on the image.
                # print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                # annotated_image = image.copy()
                return_list.append(format_landmarks(mp_hands, results, image_width, image_height))
                success, image = vidcap.read()
            vidcap.release()
        cv2.destroyAllWindows()
    return return_list

# if __name__ == '__main__':
#
#     path = '../../e2e_test_videos/video_1.mp4'
#     return_list  = get_landmarks(path)
#     json_obj = json.dumps(return_list)
#     # print(json_obj)
