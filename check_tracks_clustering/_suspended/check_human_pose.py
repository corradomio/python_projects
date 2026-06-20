import cv2

from stdlib import jsonx
from human.pose_estimation import HumanPose, PoseKeypoints


def main():
    # image_path = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-09\20260609_105432\0_1_DONE\random_crop\20260609_105507_crop_no_margin.jpg"
    # image_path =  r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\tmp_result\2026-06-09\20260609_105432\0_3_DONE\random_crop\20260609_111019_crop_no_margin.jpg"
    # image_path = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\.clusters_tracks\20260513_155653\track_11\0_5803_DONE\20260608_103448_crop_no_margin.jpg"
    image_path = r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\.clusters_tracks\20260513_155653\track_15\2_8511_DONE\20260608_110016_crop_no_margin.jpg"

    pose: PoseKeypoints = HumanPose.pose(image_path,  model_name="yolo11n-pose")

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose.draw(image, threshold=0.7)

    cv2.imwrite("test.png", image)

    jsonx.dump(pose.as_dict(), "keypoints.json", )
    pass

if __name__ == "__main__":
    main()
