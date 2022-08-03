from cv2 import VideoCapture, imwrite, CAP_PROP_POS_MSEC
import os
from os import listdir, remove, mkdir
from re import finditer


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        mkdir(path)


def setup_season(season):
    mkdir_if_not_exists("img/" + season)
    mkdir_if_not_exists("img/" + season + "/cast/")
    mkdir_if_not_exists("img/" + season + "/eps/")
    mkdir_if_not_exists("img/" + season + "/tests/")
    mkdir_if_not_exists("img/" + season + "/tests/inputs/")
    mkdir_if_not_exists("img/" + season + "/tests/labels/")
    mkdir_if_not_exists("img/" + season + "/tests/outputs/")


def cleanup_season(season):
    path = "img/" + season + "/eps/"

    [remove(path + img) for img in sorted(listdir(path))]


def sample_video(path, interval, season, endtime=10e99):
    sample_interval = interval * 1000
    endtime = endtime * 1000

    vidcap = VideoCapture(path)
    ep = next(finditer("E[0-9]+", path)).group(0)

    capture_time = sample_interval
    vidcap.set(CAP_PROP_POS_MSEC, capture_time)
    success, image = vidcap.read()
    while success:
        ts = str(int(capture_time / 1000)).rjust(5, "0")
        out_path = "img/%s/eps/%s%s_%s.jpg" % (season, season, ep, ts)
        imwrite(out_path, image)
        capture_time += sample_interval
        vidcap.set(CAP_PROP_POS_MSEC, capture_time)
        success, image = vidcap.read()

        if capture_time > endtime:
            break
        # print(ts, out_path)


def sample_season(season, interval):
    cleanup_season(season)
    path = "vid/" + season + "/"

    [
        sample_video(path + vid, interval, season)
        for vid in sorted(listdir(path))
    ]


# setup_season('us42')
sample_season("us42", 5)
