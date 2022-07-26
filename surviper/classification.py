# develop a classifier for the 5 Celebrity Faces Dataset
from pickle import load

# from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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

    return (model, in_encoder, out_encoder)


def predict_image(faces, model, in_encoder, out_encoder):
    testX = in_encoder.transform([face["embedding"] for face in faces])
    names = out_encoder.inverse_transform(model.predict(testX))
    return names


def predict_images(data, model, in_encoder, out_encoder):
    return [
        dict(
            img,
            **{
                "face_names": predict_image(
                    img["faces"], model, in_encoder, out_encoder
                )
            },
        )
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

        name = img_obj["face_names"][index]
        ax.text(
            face["box"][0] + face["box"][2] / 2,
            face["box"][1],
            name,
            color="white",
            horizontalalignment="center",
            size="smaller",
        )

    plt.savefig(path + img_obj["name"])
    plt.close()


# load dataset
data = load_pickle("data/season-41.obj")
model, in_encoder, out_encoder = build_model(data)

# predict
tests = predict_images(data["tests"], model, in_encoder, out_encoder)

path = "img/Season 41/tests/outputs/"
[export_img(test, path) for test in tests]
