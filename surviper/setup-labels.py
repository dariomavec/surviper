from math import sqrt, ceil
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from numpy import asarray
from pickle import dump
from os import listdir, mkdir
import os
from re import sub
import matplotlib.pyplot as plt


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        mkdir(path)


def setup_season(season):
    mkdir_if_not_exists("img/" + season)
    mkdir_if_not_exists("img/" + season + "/cast/")
    mkdir_if_not_exists("img/" + season + "/tests/")
    mkdir_if_not_exists("img/" + season + "/tests/inputs/")
    mkdir_if_not_exists("img/" + season + "/tests/labels/")
    mkdir_if_not_exists("img/" + season + "/tests/outputs/")


def save_pickle(obj, path):
    filehandler = open(path, "wb")
    dump(obj, filehandler)


def prep_image(path):
    image = Image.open(path).convert("RGB")
    pixels = asarray(image)
    name = sub(".*/|\\.[a-z]+$|(-alt)", "", path)

    if name[0:4] == "host":
        name = "host"

    return {"name": name, "image": image, "pixels": pixels}


def prep_folder(path):
    for file in sorted(listdir(path)):
        yield prep_image(path + file)


def export_tests(img_obj, path):
    file = open(path + sub("\\.png", ".txt", img_obj["name"]) + ".txt", "a")
    img_draw = img_obj["image"].copy()
    draw = ImageDraw.Draw(img_draw)

    for face in img_obj["faces"]:
        box = face["box"]
        draw.rectangle(box.tolist(), width=5)

        # get a font
        fnt = ImageFont.truetype("img/OpenSans.ttf", 40)
        draw.text(
            xy=(box[0], box[1]),
            text=face["name"],
            font=fnt,
            fill=(255, 255, 255, 128),
        )

        file.writelines(str(face["name"]) + ",\n")

    img_draw.save(path + img_obj["name"] + ".png")
    file.close()


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


def detect_faces(imgs, mtcnn, embedder):
    frames = []
    for i, img in enumerate(imgs):
        # Get cropped and prewhitened image tensor
        faces = mtcnn(img["image"])

        boxes, _ = mtcnn.detect(img["image"])

        if faces is None:
            embeddings = []
        else:
            # Calculate embedding (unsqueeze to add batch dimension)
            embeddings = [
                {
                    "name": "face" + str(idx),
                    "embedding": embedder(face.unsqueeze(0))
                    .squeeze(0)
                    .detach()
                    .numpy(),
                    "box": boxes[idx],
                }
                for idx, face in enumerate(faces)
            ]

        frames.append(
            {
                "name": img["name"],
                "faces": embeddings,
                "pixels": asarray(img["image"]),
                "image": img["image"],
            }
        )

    return frames


def prepare_season(season):
    setup_season(season)

    # If required, create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(keep_all=True)
    # Create an inception resnet (in eval mode):
    embedder = InceptionResnetV1(pretrained="vggface2").eval()

    # Datasets
    data = {
        "cast": detect_faces(
            prep_folder("img/" + season + "/cast/"), mtcnn, embedder
        )
        + detect_faces(prep_folder("img/host/"), mtcnn, embedder),
        "tests": detect_faces(
            prep_folder("img/" + season + "/tests/inputs/"), mtcnn, embedder
        ),
    }

    export_cast_grid(data["cast"], "img/" + season + "/tests/labels/")
    [
        export_tests(test, "img/" + season + "/tests/labels/")
        for test in data["tests"]
    ]

    save_pickle(data, "data/" + season + "-training.obj")


prepare_season("us42")
