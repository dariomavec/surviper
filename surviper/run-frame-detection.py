from pickle import load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from math import ceil, floor
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from os import listdir, mkdir
from datetime import datetime, date
from cv2 import VideoCapture, CAP_PROP_POS_MSEC, imwrite
from re import finditer
import numpy as np
from numpy import linalg
from PIL import ImageDraw, ImageFont

# from sklearn.metrics import accuracy_score


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        mkdir(path)


def extract_face_embeddings(image, mtcnn, embedder, return_box=False):
    # Get cropped and prewhitened image tensor
    faces = mtcnn(image)

    if faces is None:
        return [], None

    # Calculate embedding (unsqueeze to add batch dimension)
    embeddings = [
        embedder(face.unsqueeze(0)).squeeze(0).detach().numpy()
        for face in faces
    ]

    if return_box:
        boxes, _ = mtcnn.detect(image)

        return embeddings, boxes.tolist()
    else:
        return embeddings


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

    # fit one class model as initial filter to remove unknown people
    # outlier_model = OneClassSVM()
    # outlier_model.fit(trainX)

    # fit model
    face_model = SVC(kernel="linear", probability=True)
    face_model.fit(trainX, trainY)

    return {
        # "outlier": outlier_model,
        "face": face_model,
        "trainX": trainX,
        "in-encoder": in_encoder,
        "out-encoder": out_encoder,
    }


def predict_face_names(face_embeddings, model):
    # If no faces detected return empty list
    if len(face_embeddings) == 0:
        return [], []
    else:
        dist = [
            min([linalg.norm(tX - face) for tX in model["trainX"]])
            for face in face_embeddings
        ]

        testX = model["in-encoder"].transform(face_embeddings)
        prediction = model["face"].predict(testX)
        names = model["out-encoder"].inverse_transform(prediction).tolist()

        for idx, _ in enumerate(names):
            if dist[idx] > 1.2:
                names[idx] = "unknown"

        return names, dist


def test_model(data, model):
    for img in data:
        print(img["name"])

        names, _ = predict_face_names(
            [i["embedding"] for i in img["faces"]], model
        )

        for idx, face in enumerate(img["faces"]):
            face["name"] = names[idx]

    return data


def export_img(img_obj, path):
    draw = ImageDraw.Draw(img_obj["image"])
    for face in img_obj["faces"]:
        box = face["box"]
        draw.rectangle(box.tolist(), width=5)

        # get a font
        fnt = ImageFont.truetype("img/OpenSans.ttf", 40)
        draw.text(
            xy=(box[0], box[1]),
            text=face["name"],
            font=fnt,
            fill=(45, 255, 129, 255),
        )

    img_obj["image"].save(path + img_obj["name"] + ".png")


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
        if ((capture_time - sample_interval) % (100 * sample_interval)) == 0:
            print(datetime.now().strftime("%H:%M:%S"), " => ", name, "")
        if success:
            yield image, name

        capture_time += sample_interval
        vidcap.set(CAP_PROP_POS_MSEC, capture_time)
        success, image = vidcap.read()

        if capture_time > endtime:
            break


def process_episodes(season, interval, model, export_images=False):
    path = "vid/" + season + "/"

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(keep_all=True)

    # Create an inception resnet (in eval mode):
    embedder = InceptionResnetV1(pretrained="vggface2").eval()

    frames_with_face_names = []
    for file in sorted(listdir(path)):
        video = sample_video(path + file, interval, season)

        # TODO: Move to a batched approach
        for frame, name in video:
            face_embeddings = extract_face_embeddings(frame, mtcnn, embedder)
            faces, dist = predict_face_names(face_embeddings, model)

            frames_with_face_names.append(
                {
                    "file": name,
                    "faces": faces,
                }
            )
            if export_images:
                print("export")

    return frames_with_face_names


def process_training(season, interval, model, cast, n, margin=2):
    path = "vid/" + season + "/"

    # Output directories
    today = str(date.today())
    mkdir_if_not_exists("img/" + season + "/training/")
    mkdir_if_not_exists("img/" + season + "/training/" + today + "/")

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(keep_all=True)

    # Create an inception resnet (in eval mode):
    embedder = InceptionResnetV1(pretrained="vggface2").eval()

    frames_with_face_names = []
    for file in sorted(listdir(path)):
        if len(cast) == 0:
            break
        video = sample_video(path + file, interval, season)

        # TODO: Move to a batched approach
        for frame, name in video:
            if len(cast) == 0:
                break
            face_embeddings, boxes = extract_face_embeddings(
                frame, mtcnn, embedder, return_box=True
            )
            faces, _ = predict_face_names(face_embeddings, model)
            for idx, face in enumerate(faces):
                if face in cast.keys():
                    dest_path = (
                        "img/"
                        + season
                        + "/training/"
                        + today
                        + "/"
                        + face
                        + "-"
                        + str(cast[face])
                        + ".png"
                    )
                    box = boxes[idx]
                    frame_size = np.shape(frame)
                    box[0] = np.maximum(floor(box[0] - margin / 2), 0)
                    box[1] = np.maximum(floor(box[1] - margin / 2), 0)
                    box[2] = np.minimum(
                        ceil(box[2] + margin / 2), frame_size[1]
                    )
                    box[3] = np.minimum(
                        ceil(box[3] + margin / 2), frame_size[0]
                    )
                    img = frame[box[1] : box[3], box[0] : box[2]]
                    imwrite(dest_path, img)
                    cast[face] += 1
                    if cast[face] >= n:
                        cast.pop(face)

    return frames_with_face_names


def generate_training(season, interval, n, margin=2):
    # load dataset
    data = load_pickle("data/" + season + "-training.obj")
    cast = {c["name"]: 0 for c in data["cast"] if c["name"] != "host"}
    cast["unknown"] = 0

    model = build_model(data)

    # Export json with name, faces
    process_training(
        season=season,
        interval=interval,
        model=model,
        cast=cast,
        n=n,
        margin=margin,
    )


# run_pipeline("us1", 1)
# run_pipeline("us2", 1)
# run_pipeline("us3", 1)
# run_pipeline("us4", 1)
# run_pipeline("us5", 1)

# Capture latest timestamp where person is captured
# Capture some images at "bad" distance
generate_training("us1", interval=3, n=3, margin=50)
