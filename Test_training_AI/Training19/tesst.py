import cv2 as cv
import numpy as np
import time
import math


WINDOW_NAME = "Camera View"
GRAY_WINDOW = "Gray"
DIFF_WINDOW = "Diff"
THRESH_WINDOW = "Threshold"

click_points = []
roi_box = None
roi_reference = None

line_points = []
line_saved = False
drawing_line = False

selecting = False
selection_start_time = None

message = ""
message_time = 0

MESSAGE_DURATION = 2.0
SELECTION_TIMEOUT = 10.0

object_count = 0
next_track_id = 0
tracks = {}

# Hướng đếm:
# "left_to_right": trái sang phải
# "right_to_left": phải sang trái
# "both": cả hai hướng
count_direction = None


def mouse_callback(event, x, y, flags, param):
    global click_points
    global selecting, selection_start_time
    global message, message_time
    global line_points, drawing_line

    if event == cv.EVENT_LBUTTONDOWN:
        if drawing_line:
            if len(line_points) < 2:
                line_points.append((x, y))

            if len(line_points) == 2:
                drawing_line = False

            return

        if roi_box is None:
            if not selecting:
                selecting = True
                click_points = [(x, y)]
                selection_start_time = time.time()
            else:
                click_points.append((x, y))

                if len(click_points) == 2:
                    selecting = False

    elif event == cv.EVENT_RBUTTONDOWN:
        if selecting:
            selecting = False
            click_points = []
            message = "Selection cancelled"
            message_time = time.time()

        elif drawing_line:
            drawing_line = False
            line_points = []
            message = "Line cancelled"
            message_time = time.time()


def get_roi_coords(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return (
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    )


def get_line_side(point, line_start, line_end):
    """
    Xác định object ở bên trái hoặc bên phải line.

    Trả về:
    -1: bên trái line
     1: bên phải line
     0: nằm trên line
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Line thẳng đứng
    if x1 == x2:
        line_x = x1

    # Line hơi nghiêng: nội suy vị trí X của line tại Y hiện tại
    elif y1 != y2:
        line_x = x1 + ((py - y1) * (x2 - x1) / (y2 - y1))

    # Line nằm ngang
    else:
        line_x = (x1 + x2) / 2

    tolerance = 3

    if px < line_x - tolerance:
        return -1

    if px > line_x + tolerance:
        return 1

    return 0


def get_centroid(x, y, w, h):
    return (x + w // 2, y + h // 2)


def distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1],
    )


def update_tracks(detections):
    global tracks, next_track_id, object_count

    matched_track_ids = set()
    matched_detection_ids = set()

    candidates = []

    for track_id, track in tracks.items():
        for detection_id, detection in enumerate(detections):
            dist = distance(
                track["center"],
                detection["center"],
            )

            candidates.append(
                (dist, track_id, detection_id)
            )

    candidates.sort()

    for dist, track_id, detection_id in candidates:
        if dist > 80:
            continue

        if track_id in matched_track_ids:
            continue

        if detection_id in matched_detection_ids:
            continue

        track = tracks[track_id]
        detection = detections[detection_id]

        previous_side = track["side"]
        current_side = detection["side"]

        crossed = (
            previous_side != 0
            and current_side != 0
            and previous_side != current_side
        )

        direction = None

        if crossed:
            if previous_side == -1 and current_side == 1:
                direction = "left_to_right"

            elif previous_side == 1 and current_side == -1:
                direction = "right_to_left"

        if crossed and (
            count_direction == "both"
            or direction == count_direction
        ):
            object_count += 1

        track["center"] = detection["center"]
        track["side"] = current_side
        track["bbox"] = detection["bbox"]
        track["missed"] = 0

        matched_track_ids.add(track_id)
        matched_detection_ids.add(detection_id)

    # Tạo track mới
    for detection_id, detection in enumerate(detections):
        if detection_id in matched_detection_ids:
            continue

        tracks[next_track_id] = {
            "center": detection["center"],
            "side": detection["side"],
            "bbox": detection["bbox"],
            "missed": 0,
        }

        next_track_id += 1

    # Xóa track không còn xuất hiện
    tracks_to_delete = []

    for track_id, track in tracks.items():
        if track_id not in matched_track_ids:
            track["missed"] += 1

            if track["missed"] > 10:
                tracks_to_delete.append(track_id)

    for track_id in tracks_to_delete:
        del tracks[track_id]


def create_empty_image():
    return np.zeros((240, 320), dtype=np.uint8)


def get_direction_text():
    if count_direction == "left_to_right":
        return "Direction: Left to Right"

    if count_direction == "right_to_left":
        return "Direction: Right to Left"

    if count_direction == "both":
        return "Direction: Both"

    return "Direction: Not selected"


def main():
    global click_points, roi_box, roi_reference
    global selecting, selection_start_time
    global message, message_time
    global line_points, line_saved, drawing_line
    global object_count, tracks, next_track_id
    global count_direction

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, mouse_callback)

    cv.namedWindow(GRAY_WINDOW)
    cv.namedWindow(DIFF_WINDOW)
    cv.namedWindow(THRESH_WINDOW)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv.flip(frame, 1)
        display = frame.copy()

        # Timeout chọn ROI
        if selecting and selection_start_time is not None:
            elapsed = time.time() - selection_start_time
            remaining = SELECTION_TIMEOUT - elapsed

            if remaining <= 0:
                selecting = False
                click_points = []
                message = "Selection timeout!"
                message_time = time.time()

            else:
                cv.putText(
                    display,
                    f"Click 2 points: {remaining:.1f}s",
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                for point in click_points:
                    cv.circle(
                        display,
                        point,
                        5,
                        (0, 0, 255),
                        -1,
                    )

        # ROI đang chọn
        if len(click_points) == 2 and roi_box is None:
            x1, y1, x2, y2 = get_roi_coords(
                click_points[0],
                click_points[1],
            )

            cv.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )

            cv.putText(
                display,
                "Press ENTER to save ROI",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        # Line đang vẽ
        if len(line_points) == 1:
            cv.circle(
                display,
                line_points[0],
                5,
                (255, 255, 0),
                -1,
            )

            cv.putText(
                display,
                "Click second point for line",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        elif len(line_points) == 2 and not line_saved:
            cv.line(
                display,
                line_points[0],
                line_points[1],
                (255, 255, 0),
                2,
            )

            cv.putText(
                display,
                "Press ENTER to save line",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

        # Thông báo
        if message:
            if time.time() - message_time < MESSAGE_DURATION:
                cv.putText(
                    display,
                    message,
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                message = ""

        detections = []

        # Xử lý ROI
        if roi_box is not None and roi_reference is not None:
            x1, y1, x2, y2 = roi_box

            cv.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            roi_frame = frame[y1:y2, x1:x2]

            if roi_frame.size > 0:
                gray = cv.cvtColor(
                    roi_frame,
                    cv.COLOR_BGR2GRAY,
                )

                blurred = cv.bilateralFilter(
                    gray,
                    27,
                    75,
                    75,
                )

                diff = cv.absdiff(
                    roi_reference,
                    blurred,
                )

                threshold = cv.threshold(
                    diff,
                    15,
                    255,
                    cv.THRESH_BINARY,
                )[1]

                threshold = cv.dilate(
                    threshold,
                    None,
                    iterations=2,
                )

                contours, _ = cv.findContours(
                    threshold,
                    cv.RETR_EXTERNAL,
                    cv.CHAIN_APPROX_SIMPLE,
                )

                for contour in contours:
                    if cv.contourArea(contour) < 500:
                        continue

                    bx, by, bw, bh = cv.boundingRect(contour)

                    center_roi = get_centroid(
                        bx,
                        by,
                        bw,
                        bh,
                    )

                    center_full = (
                        x1 + center_roi[0],
                        y1 + center_roi[1],
                    )

                    side = 0

                    if line_saved:
                        side = get_line_side(
                            center_full,
                            line_points[0],
                            line_points[1],
                        )

                    detections.append(
                        {
                            "center": center_full,
                            "side": side,
                            "bbox": (
                                x1 + bx,
                                y1 + by,
                                bw,
                                bh,
                            ),
                        }
                    )

                    cv.rectangle(
                        display,
                        (x1 + bx, y1 + by),
                        (x1 + bx + bw, y1 + by + bh),
                        (0, 0, 255),
                        2,
                    )

                    cv.putText(
                        display,
                        "Object",
                        (x1 + bx, max(15, y1 + by - 10)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

                if line_saved and count_direction is not None:
                    update_tracks(detections)

                cv.imshow(GRAY_WINDOW, gray)
                cv.imshow(DIFF_WINDOW, diff)
                cv.imshow(THRESH_WINDOW, threshold)

        else:
            cv.imshow(
                GRAY_WINDOW,
                create_empty_image(),
            )

            cv.imshow(
                DIFF_WINDOW,
                create_empty_image(),
            )

            cv.imshow(
                THRESH_WINDOW,
                create_empty_image(),
            )

        # Line đã lưu
        if line_saved:
            cv.line(
                display,
                line_points[0],
                line_points[1],
                (255, 255, 0),
                3,
            )

        # Hướng và bộ đếm
        cv.putText(
            display,
            get_direction_text(),
            (10, 130),
            cv.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )

        cv.putText(
            display,
            "1: Left->Right | 2: Right->Left | 3: Both",
            (10, 160),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv.putText(
            display,
            f"COUNT: {object_count}",
            (10, 200),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv.imshow(WINDOW_NAME, display)

        key = cv.waitKey(1) & 0xFF

        if key == 27:
            break

        # Lưu ROI hoặc line
        elif key in (13, 10):
            if len(click_points) == 2 and roi_box is None:
                x1, y1, x2, y2 = get_roi_coords(
                    click_points[0],
                    click_points[1],
                )

                if x2 - x1 > 10 and y2 - y1 > 10:
                    roi_box = (x1, y1, x2, y2)

                    roi_frame = frame[y1:y2, x1:x2]

                    gray = cv.cvtColor(
                        roi_frame,
                        cv.COLOR_BGR2GRAY,
                    )

                    roi_reference = cv.GaussianBlur(
                        gray,
                        (33, 33),
                        0,
                    )

                    click_points = []

                    message = (
                        "ROI saved! Press L to draw line"
                    )
                    message_time = time.time()

            elif len(line_points) == 2 and not line_saved:
                if count_direction is None:
                    message = "Select direction: 1, 2 or 3"
                    message_time = time.time()
                else:
                    line_saved = True
                    drawing_line = False
                    tracks.clear()

                    message = "Line saved!"
                    message_time = time.time()

        # Vẽ line
        elif key == ord("l"):
            if roi_box is not None:
                line_points = []
                line_saved = False
                drawing_line = True
                tracks.clear()

                message = "Click 2 points to draw line"
                message_time = time.time()

        # Chọn hướng
        elif key == ord("1"):
            count_direction = "left_to_right"
            tracks.clear()

            message = "Direction: Left to Right"
            message_time = time.time()

        elif key == ord("2"):
            count_direction = "right_to_left"
            tracks.clear()

            message = "Direction: Right to Left"
            message_time = time.time()

        elif key == ord("3"):
            count_direction = "both"
            tracks.clear()

            message = "Direction: Both"
            message_time = time.time()

        # Reset
        elif key == ord("r"):
            click_points = []
            roi_box = None
            roi_reference = None

            line_points = []
            line_saved = False
            drawing_line = False

            selecting = False
            selection_start_time = None

            tracks.clear()
            next_track_id = 0
            object_count = 0
            count_direction = None

            message = "Reset"
            message_time = time.time()

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()