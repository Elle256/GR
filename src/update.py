import tensorflow as tf
import copy
from typing import Dict, List, Tuple
from .nets import get_model

# Map custom model class -> (model_name, dataset) for the get_model factory.
# Extending this dict is enough if new architectures are added to nets.py.
_MODEL_KEY_MAP = {
    "MLP":      ("mlp",  "mnist"),
    "CNNMnist": ("cnn",  "mnist"),
    "CNNCifar": ("cnn",  "cifar"),
}


def _clone_model_with_weights(model: tf.keras.Model) -> tf.keras.Model:
    """
    Clone a model by re-instantiating it via the get_model() factory
    (which fully builds the model) and then copying the weights.

    This avoids two problems with tf.keras.models.clone_model():
      1. Custom tf.keras.Model subclasses have no .input_shape attribute.
      2. clone_model() does not build nested Sequential sub-models, causing
         a weights-count mismatch on set_weights().
    """
    class_name = type(model).__name__

    if class_name not in _MODEL_KEY_MAP:
        raise ValueError(
            f"_clone_model_with_weights: unknown model class '{class_name}'. "
            f"Add it to _MODEL_KEY_MAP in update.py."
        )

    model_name, dataset = _MODEL_KEY_MAP[class_name]

    local_model = get_model(model_name, dataset)   # fully built
    local_model.set_weights(model.get_weights())   # safe copy

    return local_model


class LocalUpdate:
    def __init__(self, args, dataset, idxs: List[int]):
        self.args = args

        # dataset already assumed as (x, y)
        self.dataset = dataset

        self.train_data = tf.data.Dataset.from_tensor_slices(
            (dataset[0][idxs], dataset[1][idxs])
        )

        self.train_data = (
            self.train_data
            .shuffle(1000)
            .batch(args.local_bs)
        )

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

    def train(self, model: tf.keras.Model) -> Tuple[Dict, float]:

        # clone model (safe: uses factory so all sub-models are fully built)
        local_model = _clone_model_with_weights(model)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.args.lr,
            momentum=0.9,
        )

        epoch_losses = []

        for _ in range(self.args.local_epochs):
            batch_losses = []

            for x_batch, y_batch in self.train_data:
                with tf.GradientTape() as tape:
                    logits = local_model(x_batch, training=True)
                    loss = self.loss_fn(y_batch, logits)

                grads = tape.gradient(loss, local_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, local_model.trainable_variables))

                batch_losses.append(loss.numpy())

            epoch_losses.append(sum(batch_losses) / len(batch_losses))

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        return local_model.get_weights(), avg_loss


# FedProx - local SGD with proximal term
class LocalUpdateProx:
    def __init__(self, args, dataset, idxs: List[int]):
        self.args = args

        self.dataset = dataset

        self.train_data = tf.data.Dataset.from_tensor_slices(
            (dataset[0][idxs], dataset[1][idxs])
        )

        self.train_data = (
            self.train_data
            .shuffle(1000)
            .batch(args.local_bs)
        )

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True   # model outputs raw logits, not probabilities
        )

    def train(
        self,
        model: tf.keras.Model,
        global_model: tf.keras.Model,
    ) -> Tuple[Dict, float]:

        local_model = _clone_model_with_weights(model)

        global_weights = global_model.get_weights()

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.args.lr,
            momentum=0.9,
            decay=1e-4
        )

        mu = self.args.mu

        epoch_losses = []

        for epoch in range(self.args.local_epochs):
            batch_losses = []

            for x_batch, y_batch in self.train_data:

                with tf.GradientTape() as tape:
                    logits = local_model(x_batch, training=True)

                    ce_loss = self.loss_fn(y_batch, logits)

                    # FedProx term: mu/2 * ||w - w_global||^2
                    prox_loss = 0.0
                    local_vars = local_model.trainable_variables

                    for w, w_global in zip(local_vars, global_weights):
                        prox_loss += tf.reduce_sum(tf.square(w - w_global))

                    prox_loss = (mu / 2.0) * prox_loss

                    loss = ce_loss + prox_loss

                grads = tape.gradient(loss, local_vars)
                optimizer.apply_gradients(zip(grads, local_vars))

                batch_losses.append(ce_loss.numpy())

            epoch_losses.append(sum(batch_losses) / len(batch_losses))

        avg_loss = sum(epoch_losses) / len(epoch_losses)

        return local_model.get_weights(), avg_loss