"""
Neural network architectures for Federated Learning experiments.
"""

"""
TensorFlow / Keras implementations for Federated Learning experiments.

Models
------
- MLP       : 2-layer fully-connected (MNIST default)
- CNNMnist  : LeNet-style CNN for MNIST
- CNNCifar  : CNN for CIFAR-10
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# ─────────────────────────────────────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────────────────────────────────────

class MLP(tf.keras.Model):
    """
    Two-hidden-layer MLP for MNIST
    (28 × 28 grayscale → 10 classes)
    """

    def __init__(
        self,
        dim_hidden=256,
        num_classes=10,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dim_hidden = dim_hidden
        self.num_classes = num_classes

        self.model = models.Sequential([

            layers.Input(shape=(28, 28)),

            layers.Flatten(),

            layers.Dense(dim_hidden),
            layers.ReLU(),
            layers.Dropout(0.2),

            layers.Dense(dim_hidden // 2),
            layers.ReLU(),
            layers.Dropout(0.2),

            # logits output
            layers.Dense(num_classes)
        ])

    def call(self, x, training=False):

        return self.model(
            x,
            training=training
        )


# ─────────────────────────────────────────────────────────────────────────────
# CNN – MNIST
# ─────────────────────────────────────────────────────────────────────────────

class CNNMnist(tf.keras.Model):
    """
    CNN for MNIST
    (1 × 28 × 28 → 10 classes)
    """

    def __init__(
        self,
        num_classes=10,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        # feature extractor
        self.features = models.Sequential([

            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(
                32,
                kernel_size=5,
                padding='same',
                activation='relu'
            ),

            layers.MaxPooling2D(pool_size=2),

            layers.Conv2D(
                64,
                kernel_size=5,
                padding='same',
                activation='relu'
            ),

            layers.MaxPooling2D(pool_size=2),
        ])

        # classifier
        self.classifier = models.Sequential([

            layers.Flatten(),

            layers.Dense(
                512,
                activation='relu'
            ),

            layers.Dropout(0.5),

            # logits output
            layers.Dense(num_classes)
        ])

    def call(self, x, training=False):

        x = self.features(
            x,
            training=training
        )

        x = self.classifier(
            x,
            training=training
        )

        return x


# ─────────────────────────────────────────────────────────────────────────────
# CNN – CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

class CNNCifar(tf.keras.Model):
    """
    CNN for CIFAR-10
    (3 × 32 × 32 → 10 classes)
    """

    def __init__(
        self,
        num_classes=10,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.features = models.Sequential([

            # Input
            layers.Input(shape=(32, 32, 3)),

            # ─────────────────────────────
            # Block 1
            # ─────────────────────────────

            layers.Conv2D(
                64,
                kernel_size=3,
                padding='same'
            ),

            layers.BatchNormalization(),

            layers.ReLU(),

            layers.Conv2D(
                64,
                kernel_size=3,
                padding='same'
            ),

            layers.BatchNormalization(),

            layers.ReLU(),

            layers.MaxPooling2D(
                pool_size=2
            ),

            layers.Dropout(0.25),

            # ─────────────────────────────
            # Block 2
            # ─────────────────────────────

            layers.Conv2D(
                128,
                kernel_size=3,
                padding='same'
            ),

            layers.BatchNormalization(),

            layers.ReLU(),

            layers.Conv2D(
                128,
                kernel_size=3,
                padding='same'
            ),

            layers.BatchNormalization(),

            layers.ReLU(),

            layers.MaxPooling2D(
                pool_size=2
            ),

            layers.Dropout(0.25),
        ])

        self.classifier = models.Sequential([

            layers.Flatten(),

            layers.Dense(
                512,
                activation='relu'
            ),

            layers.Dropout(0.5),

            # logits output
            layers.Dense(num_classes)
        ])

    def call(self, x, training=False):

        x = self.features(
            x,
            training=training
        )

        x = self.classifier(
            x,
            training=training
        )

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(model_name: str, dataset: str):

    key = (model_name.lower(), dataset.lower())

    if key == ("mlp", "mnist"):
        model = MLP()
        dummy = tf.zeros((1, 28, 28, 1))

    elif key == ("cnn", "mnist"):
        model = CNNMnist()
        dummy = tf.zeros((1, 28, 28, 1))

    elif key in [("cnn", "cifar"), ("mlp", "cifar")]:
        model = CNNCifar()
        dummy = tf.zeros((1, 32, 32, 3))

    else:
        raise ValueError(f"Unsupported (model, dataset) pair: {key}")

    # Forward pass fully builds all nested Sequential sub-models.
    # .build() alone does NOT do this for custom tf.keras.Model subclasses.
    model(dummy, training=False)

    return model