import pandas as pd
import os
import cv2
import glob


def get_mapping():
    gesture_map = {}
    gesture_map["ASL_letter_A"] = 0
    gesture_map["ASL_letter_B"] = 1
    gesture_map["ASL_letter_C"] = 2
    gesture_map["ASL_letter_L"] = 3
    gesture_map["ASL_letter_R"] = 4
    gesture_map["ASL_letter_U"] = 5
    return gesture_map

def get_gesture_list():
    gesture_list = []
    gesture_list.append("ASL_letter_A")
    gesture_list.append("ASL_letter_B")
    gesture_list.append("ASL_letter_C")
    gesture_list.append("ASL_letter_L")
    gesture_list.append("ASL_letter_R")
    gesture_list.append("ASL_letter_U")
    return gesture_list

def concatenated_df(path="../../final_project_dataset_v0", annotations_file="annotations.csv"):
    list_directories = os.listdir(path)
    list_dfs = []

    for l in list_directories:
        if "imgs" not in l:
            df = pd.read_csv(path + "/" + l + "/" + annotations_file, index_col=None, header=0)
            list_dfs.append(df)


    all_df = pd.concat(list_dfs, axis=0, ignore_index=True)
    return all_df

def convert_video_to_images(path="../../final_project_dataset_v0", video_file_name="video"):
    list_directories = os.listdir(path)

    if not os.path.isdir(path + "/imgs"):
        os.mkdir(path + "/imgs")
        for l in list_directories:
            video_count = 0
            for video_file in glob.glob(path + "/" + l + "/videos/*.mp4"):
                pth = video_file.split("\\")
                video_name = pth[1].split(".")[0]
                vidcap = cv2.VideoCapture(video_file)
                if vidcap.isOpened():
                    success, image = vidcap.read()
                    count = 0
                    while success:
                        cv2.imwrite(
                            path + "/imgs/" + l + "$$" + video_name + "$$" + "frame_" + str(count) + "$$" + ".jpg",
                            image)
                        success, image = vidcap.read()
                        count += 1
                    video_count += 1
                    vidcap.release()
                cv2.destroyAllWindows()

    else:
        print("Directory already exists!")

# if __name__ == '__main__':
#     convert_video_to_images()