from keras_facenet import FaceNet
from PIL import Image
from numpy import asarray
from pickle import dump
from os import listdir
from re import sub
from matplotlib import pyplot


def save_pickle(obj, path):
    filehandler = open(path, "wb")
    dump(obj, filehandler)


def prep_image(path):
    image = Image.open(path).convert("RGB")
    pixels = asarray(image)

    return {
        "name": sub(".*/|\\.[a-z]+", "", path),
        "image": image,
        "pixels": pixels,
    }


def prep_folder(path):
    return [prep_image(path + file) for file in listdir(path)]


def plot_img(pixels):
    pyplot.axis("off")
    pyplot.imshow(pixels)
    pyplot.show()


def detect_faces(imgs, embedder):
    return [
        dict(p_img, **{"faces": embedder.extract(p_img["pixels"])})
        for p_img in imgs
    ]


# Gets a detection dict for each face
# in an image. Each one has the bounding box and
# face landmarks (from mtcnn.MTCNN) along with
# the embedding from FaceNet.
embedder = FaceNet()

# Datasets
data = {
    "cast": detect_faces(prep_folder("img/Season 41/cast/"), embedder),
    "tests": detect_faces(
        prep_folder("img/Season 41/tests/inputs/"), embedder
    ),
}

save_pickle(data, "data/season-41.obj")
