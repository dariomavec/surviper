from keras_facenet import FaceNet
from PIL import Image
from numpy import asarray

embedder = FaceNet()

# load image from file
image = Image.open("img/daario.jpg")
image = image.convert("RGB")
print(image)

pixels = asarray(image)

# Gets a detection dict for each face
# in an image. Each one has the bounding box and
# face landmarks (from mtcnn.MTCNN) along with
# the embedding from FaceNet.
detections = embedder.extract(pixels, threshold=0.95)

print(detections)
