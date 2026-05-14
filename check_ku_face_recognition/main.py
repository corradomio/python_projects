import json
import logging.config
import os
import cv2
from typing import Counter, Optional

import requests
import face_detection

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")


def most_common_person_name(names_list: list[str]):
    if names_list is None or len(names_list) == 0:
        return "NO_FACES_SAVED"
    ignores = [ "face_not_in_DB", "NO_FACES_SAVED"]
    filtered_names = [name for name in names_list if name not in ignores]
    if not filtered_names or len(filtered_names) == 0:
        if "face_not_in_DB" in names_list:
            return "face_not_in_DB"
        else:
            return "NO_FACES_SAVED"
    counts = Counter(filtered_names)
    name, count = counts.most_common(1)[0]
    return name


class KUFaceServer:
    def __init__(self, min_threshold=0.80):

        self.KU_FACE_API_URL = "https://faceapi.ku.ac.ae/api/OFace/MatchFaces"
        self.payload = {
            "faceKey": "LAB001 ",
            "threshold": 0.95
        }
        self.headers = {
            "X-API-Key": "sk-oface-labmonitor-9f4b2a7e6c1d4830b5a2f9e6c3d4b1a0"
        }
        self.available = True
        self.min_threshold = min_threshold

        self.log = logging.getLogger("KUFaceServer")
        pass

    def call_ku_api(self, this_face_img_folder):
        """
        Call KU Face Recognition API on the 'face_recognition' images for student/other_person recognition
        """
        log = self.log

        person_name = self.get_face_from_ku_api(this_face_img_folder)

        return person_name
    # end

    def get_face_from_ku_api(self, face_recognition_folder):
        log = self.log

        if not face_recognition_folder.endswith("\\") and not face_recognition_folder.endswith("/"):
            face_recognition_folder += "/"

        if not os.path.exists(face_recognition_folder):
            return "NO_FACES_SAVED"

        filenames = os.listdir(face_recognition_folder)
        list_of_names: list[Optional[str]] = []

        for filename in filenames:
            if not filename.lower().endswith(IMAGE_TYPES):
                continue  # 跳过非图片文件

            img_file_path = face_recognition_folder + filename

            face_file_path = self._extract_face(img_file_path)

            try:
                # WARN: IF there is an exception in the API request, this
                #       BREAKS the entire postprocessing!!!!
                # with open(face_file_path, "rb") as img_file:
                #     files = {
                #         "imageFile": (face_file_path, img_file, "image/jpeg")
                #     }
                #     response = requests.post(KU_FACE_API_URL, headers=headers, data=payload, files=files)

                response, threshold = self._call_ku_face_api_url(face_file_path)

            except Exception as e:
                log.exception("Unable to call KU Face Recognition API")
                continue

            try:
                #   {
                #       "labAccessData": [
                #           {
                #               "user": {
                #                   "email": "100060593@ku.ac.ae",
                #                   "personId": "100060593",
                #                   "FullName": "Yazan  Hani Mousa  Abuhasheesh",
                #                   "personType": "Student",
                #                   "personCategory": "Doctorate",
                #                   "personLevel": "PH",
                #                   "jobDescription": "PhD in Engineering",
                #                   "adAccount": "100060593",
                #                   "gender": "M",
                #                   "dateOfBirth": "1999-03-26",
                #                   "nationality": "Jordan"
                #               },
                #               "courses": ...
                #               ...
                #           },
                #           ...
                #       ]
                #   }
                #
                # Note: the JSON can contain multiple users!
                #
                #  {"error": "Face matching failed", "message": "Face matching failed - no face detected or confidence too low"}
                #
                # DEBUG
                # if dump_answer:
                #     log.info(json.dumps(data))
                #     dump_answer = False
                # # end

                data = json.loads(response.text)

                # cleanup data
                if "labAccessData" in data:
                    labAccessData = data["labAccessData"]
                    for lac in labAccessData:
                        if "courses" in lac:
                            del lac["courses"]
                        lac["threshold"] = threshold

                print(json.dumps(data))

            except Exception as e:
                if response is not None:
                    log.exception(f"Unable to parse JSON: '{response.text}' using image {img_file_path}")
                else:
                    log.exception(f"Unable to call KU Face Recognition API")
                continue

            if "labAccessData" in data.keys():
                lab_access_data = data["labAccessData"][0]  # len() = 1
                user_data = lab_access_data["user"]
                user_data["threshold"] = threshold

                # print(type(use_data), use_data.keys()) #  dict_keys(["email", "personId", "FullName", "personType", "personCategory", "personLevel", "jobDescription", "adAccount", "gender", "dateOfBirth", "nationality"])
                person_id, person_name = user_data["personId"], user_data["FullName"]
                # print(use_data["personId"], use_data["FullName"])

            elif "error" in data.keys():
                person_id, person_name = None, "face_not_in_DB"
            else:
                person_id, person_name = None, None

            list_of_names.append(person_name)
        # end
        assigned_name = most_common_person_name(list_of_names)
        return assigned_name

    def _call_ku_face_api_url(self, img_file_path: str) -> tuple[requests.Response, float]:
        KU_FACE_API_URL = self.KU_FACE_API_URL
        headers = self.headers
        payload = {} | self.payload
        min_threshold = self.min_threshold

        data = {"error": "Face matching failed", "message": "Face matching failed - no face detected or confidence too low"}

        while "error" in data and payload["threshold"] > min_threshold:
            with open(img_file_path, "rb") as img_file:
                files = {
                    "imageFile": (img_file_path, img_file, "image/jpeg")
                }
                response = requests.post(KU_FACE_API_URL, headers=headers, data=payload, files=files)

            try:
                data = json.loads(response.text)
                if not "error" in data:
                    break
                else:
                    payload["threshold"] -= 0.10
            except Exception as e:
                break
        # end
        return response, payload["threshold"]
    # end

    def _extract_face(self, img_file_path):
        detector = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
        # BGR to RGB
        im = cv2.imread(img_file_path)

        # [N, [xmin, ymin, xmax, ymax, detection_confidence]]
        detections = detector.detect(im)
        if len(detections) == 0:
            return img_file_path

        xmin, ymin, xmax, ymax = map(int, detections[0,:4])

        face = im[ymin:ymax, xmin:xmax]
        cv2.imwrite("face.jpg", face)


        return "face.jpg"




def main():
    kufs = KUFaceServer()

    kufs.get_face_from_ku_api(r"D:\Projects.ebtic\project.diwang\lab_monitoring_data\.recognized_people\Aamir Younis Raja")




if __name__ == "__main__":
    logging.config.fileConfig('logging_config_post.ini')
    log = logging.getLogger("main")

    main()

