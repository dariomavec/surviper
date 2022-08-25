from pickle import load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from facenet_pytorch import MTCNN, InceptionResnetV1
from os import listdir
from datetime import datetime
from cv2 import VideoCapture, CAP_PROP_POS_MSEC
from re import finditer

# from sklearn.metrics import accuracy_score


def extract_face_embeddings(image, mtcnn, embedder):
    # Get cropped and prewhitened image tensor
    faces = mtcnn(image)

    if faces is None:
        return []

    # Calculate embedding (unsqueeze to add batch dimension)
    embeddings = [
        embedder(face.unsqueeze(0)).squeeze(0).detach().numpy()
        for face in faces
    ]

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
        if ((capture_time - sample_interval) % (100 * sample_interval)) == 0:
            print(datetime.now().strftime("%H:%M:%S"), " => ", name, "")
        if success:
            yield image, name

        capture_time += sample_interval
        vidcap.set(CAP_PROP_POS_MSEC, capture_time)
        success, image = vidcap.read()

        if capture_time > endtime:
            break


def process_episodes(season, interval, model):
    path = "vid/" + season + "/"

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(keep_all=True, device="cuda:0")

    # Create an inception resnet (in eval mode):
    embedder = InceptionResnetV1(pretrained="vggface2").eval()

    frames_with_face_names = []
    for file in sorted(listdir(path)):
        video = sample_video(path + file, interval, season)

        # TODO: Move to a batched approach (see shorturl.at/dKNTX)
        for frame, name in video:
            face_embeddings = extract_face_embeddings(frame, mtcnn, embedder)
            frames_with_face_names.append(
                {
                    "file": name,
                    "faces": predict_face_names(face_embeddings, model),
                }
            )
    return frames_with_face_names


run_pipeline("us42", 1)
