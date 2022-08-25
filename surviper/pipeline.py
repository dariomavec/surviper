from pickle import load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import sys
import os
from keras_facenet import FaceNet
from numpy import asarray
from os import listdir
from contextlib import contextmanager
from cv2 import VideoCapture, CAP_PROP_POS_MSEC
from re import finditer

# from sklearn.metrics import accuracy_score


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def extract_face_embeddings(image, embedder):
    pixels = asarray(image)
    with suppress_stdout():
        embedded_faces = [
            face["embedding"] for face in embedder.extract(pixels)
        ]
    return embedded_faces


def load_pickle(path):
    file_pi = open(path, "rb")
    return load(file_pi)


def build_model(data):
    trainX = [castaway["faces"][0]["embedding"] for castaway in data["cast"]]
    trainY = [castaway["name"] for castaway in data["cast"]]

    # normalize input vectors
    in_encoder = Normalizer(norm="l2")
    trainX = in_encoder.transform(trainX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainY)
    trainY = out_encoder.transform(trainY)

    # fit model
    model = SVC(kernel="linear", probability=True)
    model.fit(trainX, trainY)

    return {
        "model": model,
        "in-encoder": in_encoder,
        "out-encoder": out_encoder,
    }


def predict_face_names(face_embeddings, model):
    # If no faces detected return empty list
    if len(face_embeddings) == 0:
        return []
    else:
        testX = model["in-encoder"].transform(face_embeddings)
        prediction = model["model"].predict(testX)
        # prediction_prob = model["model"].predict_proba(testX)

        # TODO: Have a way to identify low likelihood predictions and cull
        names = model["out-encoder"].inverse_transform(prediction).tolist()

        return names


def test_model(data, model):
    for img in data:
        print(img["name"])
        img.update({"faces": predict_face_names(img["faces"], model)})

    return data


def export_img(img_obj, path):
    plt.axis("off")
    plt.imshow(img_obj["pixels"])
    # Get the current reference
    ax = plt.gca()

    for index, face in enumerate(img_obj["faces"]):
        ax.add_patch(
            Rectangle(
                (face["box"][0:2]),
                face["box"][2],
                face["box"][3],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
        )
        ax.text(
            face["box"][0] + face["box"][2] / 2,
            face["box"][1],
            face["name"],
            color="white",
            horizontalalignment="center",
            size="smaller",
        )

    plt.savefig(path + img_obj["name"])
    plt.close()


def run_pipeline(season, interval):
    # load dataset
    data = load_pickle("data/" + season + "-training.obj")
    model = build_model(data)

    # test
    tests = test_model(data["tests"], model)

    path = "img/" + season + "/tests/outputs/"
    [export_img(test, path) for test in tests]

    # Export json with name, faces
    episodes = process_episodes(season, interval, model)
    export = [scene for scene in episodes if len(scene["faces"]) > 0]

    with open("data/json/" + season + ".json", "w") as write:
        json.dump(export, write)


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
        name = "%s%s_%s" % (season, ep, ts)
        if capture_time % (100 * interval) == 0:
            print(name)
        if success:
            yield image, name

        capture_time += sample_interval
        vidcap.set(CAP_PROP_POS_MSEC, capture_time)
        success, image = vidcap.read()

        if capture_time > endtime:
            break


def process_episodes(season, interval, model):
    path = "vid/" + season + "/"
    # Gets a detection dict for each face
    # in an image. Each one has the bounding box and
    # face landmarks (from mtcnn.MTCNN) along with
    # the embedding from FaceNet.
    embedder = FaceNet()

    frames_with_face_names = []
    for file in sorted(listdir(path)):
        video = sample_video(path + file, interval, season)

        for frame, name in video:
            face_embeddings = extract_face_embeddings(frame, embedder)
            frames_with_face_names.append(
                {
                    "file": name,
                    "faces": predict_face_names(face_embeddings, model),
                }
            )
    return frames_with_face_names


run_pipeline("us42", 1)
