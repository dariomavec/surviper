from math import sqrt, ceil
from keras_facenet import FaceNet
from PIL import Image
from numpy import asarray
from pickle import dump
from os import listdir
from re import sub
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def save_pickle(obj, path):
    filehandler = open(path, "wb")
    dump(obj, filehandler)


def prep_image(path):
    image = Image.open(path).convert("RGB")
    pixels = asarray(image)
    name = sub(".*/|\\.[a-z]+", "", path)

    if name[0:4] == "host":
        name = "host"

    return {
        "name": name,
        "image": image,
        "pixels": pixels,
    }


def prep_folder(path):
    return [prep_image(path + file) for file in sorted(listdir(path))]


def export_tests(img_obj, path):
    plt.axis("off")
    plt.imshow(img_obj["pixels"])
    # Get the current reference
    ax = plt.gca()
    file = open(path + sub("\\.png", ".txt", img_obj["name"]) + ".txt", "a")

    for face in img_obj["faces"]:
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

        file.writelines(str(face["name"]) + ",\n")

    file.close()
    plt.savefig(path + img_obj["name"])
    plt.close()


def export_cast_grid(cast, path):
    grid_size = ceil(sqrt(len(cast)))

    file = open(path + "cast.txt", "w")

    # enumerate files
    for idx, contestant in enumerate(cast):
        plt.subplot(grid_size, grid_size, idx + 1)
        plt.axis("off")
        plt.imshow(contestant["pixels"])
        # Get the current reference
        ax = plt.gca()
        ax.text(
            0,
            0,
            contestant["name"],
            color="black",
            horizontalalignment="left",
            size="smaller",
        )

        file.writelines(contestant["name"] + "\n")

    plt.savefig(path + "cast.png")
    plt.close()
    file.close()


def detect_faces(imgs, embedder):
    img_obj = [
        dict(
            p_img,
            **{
                "faces": [
                    dict(embedding, **{"name": idx})
                    for idx, embedding in enumerate(
                        embedder.extract(p_img["pixels"])
                    )
                ]
            },
        )
        for p_img in imgs
    ]

    return img_obj


def prepare_season(season):
    # Gets a detection dict for each face
    # in an image. Each one has the bounding box and
    # face landmarks (from mtcnn.MTCNN) along with
    # the embedding from FaceNet.
    embedder = FaceNet()

    # Datasets
    data = {
        "cast": detect_faces(
            prep_folder("img/" + season + "/cast/") + prep_folder("img/host/"),
            embedder,
        ),
        "tests": detect_faces(
            prep_folder("img/" + season + "/tests/inputs/"), embedder
        ),
    }

    export_cast_grid(data["cast"], "img/" + season + "/tests/labels/")
    [
        export_tests(test, "img/" + season + "/tests/labels/")
        for test in data["tests"]
    ]

    save_pickle(data, "data/" + season + "-training.obj")


def process_episodes(season):
    # Gets a detection dict for each face
    # in an image. Each one has the bounding box and
    # face landmarks (from mtcnn.MTCNN) along with
    # the embedding from FaceNet.
    embedder = FaceNet()

    # Datasets
    data = detect_faces(prep_folder("img/" + season + "/eps/"), embedder)

    save_pickle(data, "data/" + season + "-episodes.obj")


prepare_season("us42")
# process_episodes("us42")
