from PIL import Image
import tensorflow as tf
import numpy as np

model = model = tf.keras.models.load_model("data/model/Malte.h5")


img_path = tf.data.Dataset.from_tensor_slices(["data/frames/000_0000.png",
                                            "data/frames/000_0100.png",
                                            "data/frames/000_0030.png",
                                            "data/frames/001_0030.png",
                                            "data/frames/002_0030.png"])

def get_images(dataset):
    images = []
    for data in dataset:
        filename = data.numpy()
        image = Image.open(filename)
        images.append(np.array(image))
    images = np.array(images)/255
    return images

def output_to_image(output):
    output = output * 255
    images = []
    for img in output:
        images.append(Image.fromarray(img.astype(np.uint8)))
    return images

images = get_images(img_path)
bg = model.predict(images)

bg = output_to_image(bg)
bg[3].show()
im = Image.open("data/frames/000_0000.png")
print(im)
im.show()
