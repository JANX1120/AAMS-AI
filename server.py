from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

import numpy as np
import cv2


from detector import detect_frame


# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(
    title="AAMS AI Detection API"
)


app.add_middleware(

    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=False,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


# =====================================================
# HOME
# =====================================================

@app.get("/")
def home():

    return {

        "success":
            True,

        "message":
            "AAMS AI Detection Server is running.",

        "models": {

            "object_detection":
                "YOLO",

            "head_pose":
                "6DRepNet"
        }
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {

        "success":
            True,

        "status":
            "online"
    }


# =====================================================
# AI DETECTION ENDPOINT
# =====================================================

@app.post("/detect")
async def detect(

    file: UploadFile = File(...)

):

    try:

        # =====================================================
        # READ UPLOADED IMAGE
        # =====================================================

        image_bytes = (
            await file.read()
        )


        if not image_bytes:

            return {

                "success":
                    False,

                "message":
                    "No image data was received."
            }


        # =====================================================
        # CONVERT TO NUMPY
        # =====================================================

        np_array = np.frombuffer(

            image_bytes,

            np.uint8
        )


        # =====================================================
        # DECODE IMAGE
        # =====================================================

        frame = cv2.imdecode(

            np_array,

            cv2.IMREAD_COLOR
        )


        if frame is None:

            return {

                "success":
                    False,

                "message":
                    "Invalid image."
            }


        # =====================================================
        # AI DETECTION
        # =====================================================

        result = detect_frame(
            frame
        )


        return result


    except Exception as error:

        print(
            "Detection error:",
            error
        )


        return {

            "success":
                False,

            "message":
                str(error)
        }