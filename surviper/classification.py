# develop a classifier for the 5 Celebrity Faces Dataset
from pickle import load

# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json


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


def predict_faces(faces, model):
    # If no faces detected return empty list
    if len(faces) == 0:
        return []
    else:
        testX = model["in-encoder"].transform(
            [face["embedding"] for face in faces]
        )

        prediction = model["model"].predict(testX)

        # TODO: Have a way to identify low likelihood predictions and cull
        names = model["out-encoder"].inverse_transform(prediction)
        for i, name in enumerate(names):
            faces[i].update({"name": name})
        return faces


def run_model(data, model):
    return [
        dict(img, **{"faces": predict_faces(img["faces"], model)})
        for img in data
    ]


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


# load dataset
data = load_pickle("data/us41-training.obj")
model = build_model(data)

# test
tests = run_model(data["tests"], model)

path = "img/us41/tests/outputs/"
[export_img(test, path) for test in tests]

# run against episodes
episodes = run_model(load_pickle("data/us41-episodes.obj"), model)

# Export json with name, faces
export = [
    {"file": scene["name"], "faces": [face["name"] for face in scene["faces"]]}
    for scene in episodes
    if len(scene["faces"]) > 0
]

with open("img/us41/episodes.json", "w") as write:
    json.dump(export, write)
