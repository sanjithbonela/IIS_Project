import requests

if __name__ == '__main__':

    URL = ''
    video_path = ""
    gesture_name = e2e_test.test_video(video_path=video_path)
    data = {'gesture_name' : gesture_name}
    resp = requests.post(URL, data)
