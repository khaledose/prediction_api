import tensorflow as tf
import numpy as np
import os

class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(Conv, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        
    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x

def load(f, label):
    # load the file into tensor
    image = tf.io.read_file(f)
    # Decode it to JPEG format
    image = tf.image.decode_jpeg(image)
    # Convert it to tf.float32
    image = tf.cast(image, tf.float32)
    
    return image, label

def load_image_train(image_file, label):
    image, label = load(image_file, label)
    image = random_jitter(image)
    image = normalize(image)
    return image, label

def load_image_val(image_file, label):
    image, label = load(image_file, label)
    image = central_crop(image)
    image = normalize(image)
    return image, label

def resize(input_image, size):
    return tf.image.resize(input_image, size)

def random_crop(input_image):
    return tf.image.random_crop(input_image, size=[150, 150, 3])

def central_crop(input_image):
    image = resize(input_image, [176, 176])
    return tf.image.central_crop(image, central_fraction=0.84)

def random_rotation(input_image):
    angles = np.random.randint(0, 3, 1)
    return tf.image.rot90(input_image, k=angles[0])

def random_jitter(input_image):
    # Resize it to 176 x 176 x 3
    image = resize(input_image, [176, 176])
    # Randomly Crop to 150 x 150 x 3
    image = random_crop(image)
    # Randomly rotation
    image = random_rotation(image)
    # Randomly mirroring
    image = tf.image.random_flip_left_right(image)
    return image

def normalize(input_image):
    mid = (tf.reduce_max(input_image) + tf.reduce_min(input_image)) / 2
    input_image = input_image / mid - 1
    return input_image



def setup_model(path):
    checkpoint_path = os.path.join(path, "cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model = tf.keras.Sequential(name='CNN')

    model.add(Conv(filters=32, kernel_size=(3, 3)))
    model.add(Conv(filters=64, kernel_size=(3, 3)))
    model.add(Conv(filters=128, kernel_size=(3, 3)))
    model.add(Conv(filters=128, kernel_size=(3, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax))

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), 
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
                    metrics = ['accuracy'])

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    return model

def run_prediction(model, img):
    im, _ = load_image_val(img, 0)
    im = tf.reshape(im, (1,148,148,3))
    im.shape
    pred = model.predict(im)
    return np.argmax(pred) 