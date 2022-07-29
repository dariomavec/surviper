from cv2 import VideoCapture, imwrite, CAP_PROP_POS_MSEC
from os import listdir, remove


def cleanup_season(season):
    path = "img/" + season + "/eps/"

    [remove(path + vid) for vid in sorted(listdir(path))]


def sample_video(path, interval):
    sample_interval = interval * 1000

    vidcap = VideoCapture(path)

    capture_time = sample_interval
    vidcap.set(CAP_PROP_POS_MSEC, capture_time)
    success, image = vidcap.read()
    while success:
        ts = str(int(capture_time / 1000)).rjust(5, "0")
        imwrite("img/us41/eps/US41E01_%s.jpg" % ts, image)
        capture_time += sample_interval
        vidcap.set(CAP_PROP_POS_MSEC, capture_time)
        success, image = vidcap.read()


def sample_season(season, interval):
    cleanup_season(season)
    path = "vid/" + season + "/"

    [sample_video(path + vid, interval) for vid in sorted(listdir(path))]


sample_season("us41", 5)
