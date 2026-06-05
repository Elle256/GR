import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple


def evaluate_global(
    model: tf.keras.Model,
    test_dataset,
    batch_size: int = 128,
) -> Tuple[float, float]:

    x_test, y_test = test_dataset

    test_data = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).batch(batch_size)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    )

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_data:

        logits = model(images, training=False)

        loss = loss_fn(labels, logits)

        total_loss += loss.numpy() * labels.shape[0]

        preds = tf.argmax(
            logits,
            axis=1,
            output_type=tf.int32
        )

        labels = tf.cast(labels, tf.int32)

        correct += tf.reduce_sum(
            tf.cast(preds == labels, tf.int32)
        ).numpy()

        total += labels.shape[0]

    acc = correct / total if total > 0 else 0.0

    avg_loss = total_loss / total if total > 0 else 0.0

    return acc, avg_loss


def evaluate_per_client(
    model: tf.keras.Model,
    train_dataset,
    client_idxs: Dict[int, List[int]],
    selected_clients: List[int],
    batch_size: int = 64,
) -> List[float]:

    x_train, y_train = train_dataset

    per_client_accs = []

    for cid in selected_clients:

        idxs = client_idxs[cid]

        if len(idxs) == 0:
            per_client_accs.append(0.0)
            continue

        client_data = tf.data.Dataset.from_tensor_slices(
            (
                x_train[idxs],
                y_train[idxs]
            )
        ).batch(batch_size)

        correct = 0
        total = 0

        for images, labels in client_data:

            logits = model(images, training=False)

            preds = tf.argmax(
                logits,
                axis=1,
                output_type=tf.int32
            )

            labels = tf.cast(labels, tf.int32)

            correct += tf.reduce_sum(
                tf.cast(preds == labels, tf.int32)
            ).numpy()

            total += labels.shape[0]

        acc = correct / total if total > 0 else 0.0

        per_client_accs.append(acc)

    return per_client_accs