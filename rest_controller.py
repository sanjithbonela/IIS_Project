import requests
from end_to_end import e2e_test

if __name__ == '__main__':

    URL = 'http://localhost:8000'
    video_path = "pred_multiple_videos('../../final_project_dataset_v1/ASL_letter_C/videos/*.mp4')"
    gesture_name = e2e_test.pred_multiple_videos('../../final_project_dataset_v1/ASL_letter_C/videos/*.mp4')
    data = {'gesture_name' : gesture_name}
    resp = requests.post(URL, data)
    print(resp)
