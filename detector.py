from ultralytics import YOLO
from sixdrepnet import SixDRepNet

import numpy as np
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = BASE_DIR / "yolo26n.pt"


# =====================================================
# LOAD AI MODELS
# =====================================================

yolo_model = YOLO(str(YOLO_MODEL_PATH))

# TEMPORARY: disabled for Render memory test
head_pose_model = None

# =====================================================
# AI SETTINGS
# =====================================================
YOLO_CONFIDENCE = 0.60

HEAD_YAW_THRESHOLD = 35
HEAD_PITCH_THRESHOLD = 30

# Ignore very weak detections even if YOLO returns them
MIN_OBJECT_CONFIDENCE = 0.60

# Minimum person box size for head-pose analysis
MIN_PERSON_WIDTH = 80
MIN_PERSON_HEIGHT = 120

# =====================================================
# HEAD DIRECTION
# =====================================================

def get_head_direction(
    pitch,
    yaw,
    roll
):

    pitch = float(pitch)
    yaw = float(yaw)
    roll = float(roll)


    if yaw <= -HEAD_YAW_THRESHOLD:
        return "left"


    if yaw >= HEAD_YAW_THRESHOLD:
        return "right"


    if pitch <= -HEAD_PITCH_THRESHOLD:
        return "up"


    if pitch >= HEAD_PITCH_THRESHOLD:
        return "down"


    return "forward"


# =====================================================
# HEAD POSE DETECTION
# =====================================================

def detect_head_pose(
    frame,
    person_boxes
):

    head_results = []


    frame_height, frame_width = (
        frame.shape[:2]
    )


    for index, person_box in enumerate(
        person_boxes
    ):

        x1 = max(
            0,
            int(person_box["x1"])
        )

        y1 = max(
            0,
            int(person_box["y1"])
        )

        x2 = min(
            frame_width,
            int(person_box["x2"])
        )

        y2 = min(
            frame_height,
            int(person_box["y2"])
        )


        person_width = x2 - x1
        person_height = y2 - y1


        if (
            person_width <= 0 or
            person_height <= 0
        ):
            continue


        # =====================================================
        # APPROXIMATE HEAD REGION
        # =====================================================

        head_x1 = (
            x1 +
            int(person_width * 0.15)
        )

        head_x2 = (
            x2 -
            int(person_width * 0.15)
        )


        head_y1 = y1

        head_y2 = (
            y1 +
            int(person_height * 0.40)
        )


        head_x1 = max(
            0,
            head_x1
        )

        head_y1 = max(
            0,
            head_y1
        )

        head_x2 = min(
            frame_width,
            head_x2
        )

        head_y2 = min(
            frame_height,
            head_y2
        )


        head_crop = frame[
            head_y1:head_y2,
            head_x1:head_x2
        ]


        if head_crop.size == 0:
            continue


        if (
            head_crop.shape[0] < 50 or
            head_crop.shape[1] < 50
        ):
            continue


        try:

            pitch, yaw, roll = (
                head_pose_model.predict(
                    head_crop
                )
            )


            pitch = float(
                np.asarray(
                    pitch
                ).reshape(-1)[0]
            )

            yaw = float(
                np.asarray(
                    yaw
                ).reshape(-1)[0]
            )

            roll = float(
                np.asarray(
                    roll
                ).reshape(-1)[0]
            )


            direction = (
                get_head_direction(
                    pitch,
                    yaw,
                    roll
                )
            )


            possible_peeking = (
                direction == "left" or
                direction == "right"
            )


            head_results.append({

                "person":
                    f"Person {index + 1}",

                "pitch":
                    round(
                        pitch,
                        2
                    ),

                "yaw":
                    round(
                        yaw,
                        2
                    ),

                "roll":
                    round(
                        roll,
                        2
                    ),

                "direction":
                    direction,

                "possible_peeking":
                    possible_peeking,

                "box": {

                    "x1":
                        head_x1,

                    "y1":
                        head_y1,

                    "x2":
                        head_x2,

                    "y2":
                        head_y2
                }
            })


        except Exception as error:

            print(
                f"Head pose error for Person {index + 1}:",
                error
            )


    return head_results


# =====================================================
# MAIN AI DETECTION
# =====================================================

def detect_frame(frame):

    # =====================================================
    # YOLO DETECTION
    # =====================================================

    results = yolo_model.predict(

        source=frame,

        conf=YOLO_CONFIDENCE,

        verbose=False
    )


    result = results[0]


    people = 0
    phones = 0
    books = 0


    detection_list = []

    person_boxes = []


    # =====================================================
    # LOOP DETECTIONS
    # =====================================================

    for box in result.boxes:

        class_id = int(
            box.cls[0]
        )


        class_name = (
            yolo_model.names[
                class_id
            ]
        )


        confidence = float(
            box.conf[0]
        )


        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )


        detection_data = {

            "class":
                class_name,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "box": {

                "x1":
                    round(x1),

                "y1":
                    round(y1),

                "x2":
                    round(x2),

                "y2":
                    round(y2)
            }
        }


        detection_list.append(
            detection_data
        )


        # =====================================================
        # PERSON
        # =====================================================

        if class_name == "person":

            people += 1


            person_boxes.append({

                "x1": x1,

                "y1": y1,

                "x2": x2,

                "y2": y2
            })


        # =====================================================
        # PHONE
        # =====================================================

        elif class_name == "cell phone":

            phones += 1


        # =====================================================
        # BOOK
        # =====================================================

        elif class_name == "book":

            books += 1


    # =====================================================
    # HEAD POSE
    # =====================================================

    head_pose_results = []

    # =====================================================
    # PEEKING INFORMATION
    # =====================================================

    peeking_count = 0

    peeking_people = []


    for head in head_pose_results:

        if head[
            "possible_peeking"
        ]:

            peeking_count += 1

            peeking_people.append(
                head["person"]
            )


    possible_peeking = (
        peeking_count > 0
    )


    # =====================================================
    # POSSIBLE VIOLATION RULES
    # =====================================================

    violation_list = []


    if phones > 0:

        violation_list.append(
            "Mobile Phone Detected"
        )


    if books > 0:

        violation_list.append(
            "Unauthorized Book Detected"
        )


    if people > 1:

        violation_list.append(
            "Multiple Persons Detected"
        )


    # Possible signal only.
    # Home.js should still confirm it
    # across multiple frames.

    if possible_peeking:

        violation_list.append(
            "Possible Peeking Detected"
        )


    # =====================================================
    # RESULT
    # =====================================================

    return {

        "success":
            True,


        # YOLO

        "people":
            people,

        "phones":
            phones,

        "books":
            books,


        # HEAD POSE

        "possible_peeking":
            possible_peeking,

        "peeking_count":
            peeking_count,

        "peeking_people":
            peeking_people,

        "head_pose":
            head_pose_results,


        # POSSIBLE VIOLATIONS

        "violations":
            len(
                violation_list
            ),

        "violation_list":
            violation_list,


        # DETECTIONS

        "detections":
            detection_list,


        # IMAGE SIZE

        "image_width":
            frame.shape[1],

        "image_height":
            frame.shape[0]
    }
