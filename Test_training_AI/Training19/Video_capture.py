# Training19/test_Video_capture.py

import numpy as np
import pytest

from Training19.Video_capture import main
from Training19 import Video_capture as video_capture


class FakeCamera:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class FakeCV:
    EVENT_LBUTTONDOWN = 1
    EVENT_RBUTTONDOWN = 2
    FONT_HERSHEY_SIMPLEX = 0

    COLOR_BGR2GRAY = 10
    THRESH_BINARY = 0
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 0

    def __init__(self, keys=(27,), camera_opened=True):
        self.keys = list(keys)
        self.camera = FakeCamera(camera_opened)
        self.created_windows = []
        self.destroyed = False
        self.shown_windows = []

    def VideoCapture(self, index):
        return self.camera

    def namedWindow(self, name):
        self.created_windows.append(name)

    def setMouseCallback(self, name, callback):
        pass

    def flip(self, frame, direction):
        return frame

    def putText(self, image, text, position, font, scale, color, thickness):
        return image

    def rectangle(self, *args, **kwargs):
        pass

    def circle(self, *args, **kwargs):
        pass

    def line(self, *args, **kwargs):
        pass

    def imshow(self, name, image):
        self.shown_windows.append(name)

    def waitKey(self, delay):
        if self.keys:
            return self.keys.pop(0)
        return 27

    def destroyAllWindows(self):
        self.destroyed = True

    def cvtColor(self, image, code):
        return np.zeros(image.shape[:2], dtype=np.uint8)

    def GaussianBlur(self, image, kernel, sigma):
        return image

    def absdiff(self, image1, image2):
        return image1

    def threshold(self, image, threshold, maximum, mode):
        return 0, image

    def dilate(self, image, kernel, iterations):
        return image

    def findContours(self, image, mode, method):
        return [], None

    def contourArea(self, contour):
        return 0

    def boundingRect(self, contour):
        return 0, 0, 0, 0


@pytest.fixture
def reset_state(monkeypatch):
    video_capture.click_points = []
    video_capture.roi_box = None
    video_capture.roi_reference = None
    video_capture.line_points = []
    video_capture.line_saved = False
    video_capture.drawing_line = False
    video_capture.selecting = False
    video_capture.selection_start_time = None
    video_capture.message = ""
    video_capture.message_time = 0
    video_capture.object_count = 0
    video_capture.next_track_id = 0
    video_capture.tracks = {}

    yield

    video_capture.click_points = []
    video_capture.roi_box = None
    video_capture.roi_reference = None
    video_capture.line_points = []
    video_capture.line_saved = False
    video_capture.drawing_line = False
    video_capture.selecting = False
    video_capture.tracks = {}
    video_capture.object_count = 0
    video_capture.next_track_id = 0


def test_main_returns_when_camera_cannot_open(monkeypatch, reset_state):
    fake_cv = FakeCV(camera_opened=False)
    monkeypatch.setattr(video_capture, "cv", fake_cv)

    main()

    assert fake_cv.camera.released is False
    assert fake_cv.destroyed is False


def test_main_releases_camera_and_closes_windows_on_escape(
    monkeypatch,
    reset_state,
):
    fake_cv = FakeCV(keys=(27,))
    monkeypatch.setattr(video_capture, "cv", fake_cv)

    main()

    assert fake_cv.camera.released is True
    assert fake_cv.destroyed is True

    assert "Camera View" in fake_cv.created_windows
    assert "Gray" in fake_cv.created_windows
    assert "Diff" in fake_cv.created_windows
    assert "Threshold" in fake_cv.created_windows


def test_main_displays_gray_diff_and_threshold_windows(
    monkeypatch,
    reset_state,
):
    fake_cv = FakeCV(keys=(27,))
    monkeypatch.setattr(video_capture, "cv", fake_cv)

    video_capture.roi_box = (0, 0, 50, 50)
    video_capture.roi_reference = np.zeros((50, 50), dtype=np.uint8)

    main()

    assert "Gray" in fake_cv.shown_windows
    assert "Diff" in fake_cv.shown_windows
    assert "Threshold" in fake_cv.shown_windows


def test_main_updates_tracking_only_after_line_is_saved(
    monkeypatch,
    reset_state,
):
    fake_cv = FakeCV(keys=(27,))
    monkeypatch.setattr(video_capture, "cv", fake_cv)

    video_capture.roi_box = (0, 0, 50, 50)
    video_capture.roi_reference = np.zeros((50, 50), dtype=np.uint8)
    video_capture.line_points = [(10, 10), (40, 40)]
    video_capture.line_saved = True

    update_calls = []

    def fake_update_tracks(detections):
        update_calls.append(detections)

    monkeypatch.setattr(
        video_capture,
        "update_tracks",
        fake_update_tracks,
    )

    main()

    assert len(update_calls) == 1
    assert update_calls[0] == []


def test_main_reset_key_clears_roi_line_tracks_and_count(
    monkeypatch,
    reset_state,
):
    fake_cv = FakeCV(keys=(ord("r"), 27))
    monkeypatch.setattr(video_capture, "cv", fake_cv)

    video_capture.roi_box = (1, 2, 30, 40)
    video_capture.roi_reference = np.zeros((38, 29), dtype=np.uint8)
    video_capture.line_points = [(5, 5), (20, 20)]
    video_capture.line_saved = True
    video_capture.tracks = {1: {"center": (10, 10)}}
    video_capture.object_count = 7
    video_capture.next_track_id = 8

    main()

    assert video_capture.roi_box is None
    assert video_capture.roi_reference is None
    assert video_capture.line_points == []
    assert video_capture.line_saved is False
    assert video_capture.tracks == {}
    assert video_capture.object_count == 0
    assert video_capture.next_track_id == 0