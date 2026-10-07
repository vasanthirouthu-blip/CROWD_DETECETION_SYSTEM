from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import cv2
import numpy as np
import onnxruntime as ort
import sqlite3
import time
import os


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# Secret key for login session
app.secret_key = "crowd_detection_secret_key"


# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "yolo11n.onnx"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "crowd_data.db"
)


# ==========================================
# YOLO ONNX MODEL
# ==========================================

session_onnx = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session_onnx.get_inputs()[0].name


# ==========================================
# CROWD SETTINGS
# ==========================================

LIMIT = 3

CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.45


# ==========================================
# LATEST DETECTION
# ==========================================

latest_count = 0
latest_status = "NORMAL"

last_saved_count = -1
last_saved_status = ""
last_saved_time = 0


# ==========================================
# LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        )

        password = request.form.get(
            "password",
            ""
        )

        # Demo login credentials
        if (
            username == "admin"
            and password == "admin123"
        ):

            session["logged_in"] = True

            return redirect(
                url_for("index")
            )

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template(
        "login.html"
    )


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ==========================================
# DATABASE
# ==========================================

def save_crowd_data(
    count,
    status
):

    try:

        connection = sqlite3.connect(
            DATABASE_PATH
        )

        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO crowd_logs
            (
                timestamp,
                people_count,
                status
            )
            VALUES
            (
                datetime(
                    'now',
                    'localtime'
                ),
                ?,
                ?
            )
        """, (
            count,
            status
        ))

        connection.commit()

        connection.close()

        print(
            "Database saved:",
            count,
            status
        )

    except Exception as e:

        print(
            "Database error:",
            e
        )


# ==========================================
# YOLO PERSON DETECTION
# ==========================================

def detect_people(frame):

    # --------------------------------------
    # Original image dimensions
    # --------------------------------------

    image_height = frame.shape[0]
    image_width = frame.shape[1]


    # --------------------------------------
    # Resize image for YOLO
    # --------------------------------------

    image = cv2.resize(
        frame,
        (640, 640)
    )


    # --------------------------------------
    # BGR → RGB
    # --------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # Normalize
    # --------------------------------------

    image = image.astype(
        np.float32
    ) / 255.0


    # --------------------------------------
    # HWC → CHW
    # --------------------------------------

    image = np.transpose(
        image,
        (2, 0, 1)
    )


    # --------------------------------------
    # Add batch dimension
    # --------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )


    # --------------------------------------
    # Run ONNX model
    # --------------------------------------

    outputs = session_onnx.run(
        None,
        {
            input_name: image
        }
    )


    predictions = outputs[0]


    # --------------------------------------
    # Remove batch dimension
    # --------------------------------------

    if predictions.ndim == 3:

        predictions = predictions[0]


    # --------------------------------------
    # YOLO11 normally gives:
    #
    # [84, 8400]
    #
    # Convert to:
    #
    # [8400, 84]
    # --------------------------------------

    if predictions.shape[0] < predictions.shape[1]:

        predictions = predictions.T


    # --------------------------------------
    # Store detections
    # --------------------------------------

    boxes = []
    scores = []


    # --------------------------------------
    # Process every prediction
    # --------------------------------------

    for detection in predictions:

        if len(detection) < 5:
            continue


        # ----------------------------------
        # Bounding box
        # ----------------------------------

        x_center = float(
            detection[0]
        )

        y_center = float(
            detection[1]
        )

        width = float(
            detection[2]
        )

        height = float(
            detection[3]
        )


        # ----------------------------------
        # Class scores
        # ----------------------------------

        class_scores = detection[4:]


        if len(class_scores) == 0:
            continue


        # ----------------------------------
        # Find highest class
        # ----------------------------------

        class_id = int(
            np.argmax(
                class_scores
            )
        )


        confidence = float(
            class_scores[class_id]
        )


        # ----------------------------------
        # COCO class 0 = person
        # ----------------------------------

        if class_id != 0:
            continue


        # ----------------------------------
        # Confidence filtering
        # ----------------------------------

        if confidence < CONFIDENCE_THRESHOLD:
            continue


        # ----------------------------------
        # Convert coordinates
        # ----------------------------------

        x_scale = image_width / 640.0
        y_scale = image_height / 640.0


        x1 = int(
            (
                x_center
                - width / 2
            )
            * x_scale
        )


        y1 = int(
            (
                y_center
                - height / 2
            )
            * y_scale
        )


        box_width = int(
            width * x_scale
        )


        box_height = int(
            height * y_scale
        )


        # ----------------------------------
        # Keep box inside image
        # ----------------------------------

        x1 = max(
            0,
            min(
                x1,
                image_width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                image_height - 1
            )
        )


        box_width = max(
            1,
            min(
                box_width,
                image_width - x1
            )
        )


        box_height = max(
            1,
            min(
                box_height,
                image_height - y1
            )
        )


        # ----------------------------------
        # Store box and confidence
        # ----------------------------------

        boxes.append([
            x1,
            y1,
            box_width,
            box_height
        ])

        scores.append(
            confidence
        )


    # ======================================
    # NO DETECTIONS
    # ======================================

    if len(boxes) == 0:

        return 0


    # ======================================
    # NON-MAXIMUM SUPPRESSION
    # ======================================

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD
    )


    # ======================================
    # NO FINAL DETECTIONS
    # ======================================

    if indices is None:
        return 0


    if len(indices) == 0:
        return 0


    # ======================================
    # FINAL PERSON COUNT
    # ======================================

    person_count = len(indices)


    return person_count


# ==========================================
# HOME / DASHBOARD
# ==========================================

@app.route("/")
def index():

    # Login required
    if not session.get(
        "logged_in"
    ):

        return redirect(
            url_for("login")
        )


    return render_template(
        "index.html",
        limit=LIMIT
    )


# ==========================================
# AI DETECTION
# ==========================================

@app.route(
    "/detect",
    methods=["POST"]
)
def detect():

    global latest_count
    global latest_status
    global last_saved_count
    global last_saved_status
    global last_saved_time


    # --------------------------------------
    # Check login
    # --------------------------------------

    if not session.get(
        "logged_in"
    ):

        return jsonify({

            "success": False,

            "error":
                "Please login first"

        }), 401


    try:

        # ----------------------------------
        # Check image
        # ----------------------------------

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "No image received"

            })


        # ----------------------------------
        # Read image
        # ----------------------------------

        file = request.files[
            "image"
        ]

        image_bytes = file.read()


        # ----------------------------------
        # Convert bytes → NumPy
        # ----------------------------------

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )


        # ----------------------------------
        # NumPy → OpenCV image
        # ----------------------------------

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({

                "success": False,

                "error":
                    "Invalid image"

            })


        # ----------------------------------
        # Detect people
        # ----------------------------------

        person_count = detect_people(
            frame
        )


        # ----------------------------------
        # Crowd status
        # ----------------------------------

        if person_count > LIMIT:

            status = "OVER CROWDED"

        else:

            status = "NORMAL"


        # ----------------------------------
        # Update latest values
        # ----------------------------------

        latest_count = person_count

        latest_status = status


        # ----------------------------------
        # Current time
        # ----------------------------------

        current_time = time.time()


        # ----------------------------------
        # Save database
        # ----------------------------------

        if (

            person_count
            != last_saved_count

            or

            status
            != last_saved_status

            or

            current_time
            - last_saved_time
            >= 5

        ):

            save_crowd_data(
                person_count,
                status
            )


            last_saved_count = (
                person_count
            )

            last_saved_status = (
                status
            )

            last_saved_time = (
                current_time
            )


        # ----------------------------------
        # Send result to dashboard
        # ----------------------------------

        return jsonify({

            "success": True,

            "people":
                person_count,

            "status":
                status,

            "limit":
                LIMIT

        })


    except Exception as e:

        print(
            "Detection error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        })


# ==========================================
# CURRENT STATUS
# ==========================================

@app.route("/status")
def status():

    if not session.get(
        "logged_in"
    ):

        return jsonify({

            "error":
                "Please login first"

        }), 401


    return jsonify({

        "people":
            latest_count,

        "status":
            latest_status,

        "limit":
            LIMIT,

        "camera":
            True

    })


# ==========================================
# DATABASE HISTORY
# ==========================================

@app.route("/history")
def history():

    if not session.get(
        "logged_in"
    ):

        return jsonify({

            "error":
                "Please login first"

        }), 401


    try:

        connection = sqlite3.connect(
            DATABASE_PATH
        )


        cursor = connection.cursor()


        cursor.execute("""
            SELECT
                id,
                timestamp,
                people_count,
                status
            FROM crowd_logs
            ORDER BY id DESC
            LIMIT 10
        """)


        records = cursor.fetchall()


        connection.close()


        return jsonify(
            records
        )


    except Exception as e:

        print(
            "History error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        })


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )