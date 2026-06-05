# Main: Load dataset, khoi tao model, chay vong huan luyen, thu thap ket qua va luu vao file
import os
import copy
import json
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Project modules
from utils import (
    args_parser,
    set_seed,
    partition_dataset,
    compute_label_distribution,
    MetricsTracker,
    compute_client_drift,
    compute_fairness,
    weighted_average,
)

from src import (
    get_model,
    LocalUpdate,
    LocalUpdateProx,
    fedavg_aggregate,
    apply_dropout,
    evaluate_global,
    evaluate_per_client,
)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_name: str):

    if dataset_name == "mnist":

        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.mnist.load_data()

        # normalize
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

        # add channel dimension
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    elif dataset_name == "cifar":

        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.cifar10.load_data()

        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

        y_train = y_train.flatten()
        y_test = y_test.flatten()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_dataset = (x_train, y_train)
    test_dataset = (x_test, y_test)

    return train_dataset, test_dataset


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard helper
# ─────────────────────────────────────────────────────────────────────────────

def get_writer(args):

    if not args.tensorboard:
        return None

    tb_dir = os.path.join(
        args.save_dir,
        "tensorboard",
        args.exp_name
    )

    os.makedirs(tb_dir, exist_ok=True)

    return tf.summary.create_file_writer(tb_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Main FL training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(args):

    # setup
    set_seed(args.seed)

    print(f"\n{'='*60}")
    print(f"  Experiment : {args.exp_name}")
    print(f"  Algorithm  : {args.algorithm.upper()}")
    print(f"  Dataset    : {args.dataset.upper()}")
    print(f"  Model      : {args.model.upper()}")
    print(f"  Clients    : {args.num_clients}")
    print(f"  Fraction   : {args.frac}")
    print(f"  Alpha      : {'IID' if args.iid else args.alpha}")
    print(f"  Dropout    : {int(args.dropout*100)}%")

    if args.algorithm == "fedprox":
        print(f"  Mu         : {args.mu}")

    print(f"  Rounds     : {args.rounds}")
    print(f"  LocalEpoch : {args.local_epochs}")

    print(f"{'='*60}\n")

    # ─────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────

    train_dataset, test_dataset = load_dataset(
        args.dataset
    )

    client_idxs = partition_dataset(
        train_dataset,
        args.num_clients,
        iid=args.iid,
        alpha=args.alpha,
    )

    client_sizes = [
        len(client_idxs[i])
        for i in range(args.num_clients)
    ]

    # ─────────────────────────────────────────
    # Global model
    # ─────────────────────────────────────────

    global_model = get_model(
        args.model,
        args.dataset
    )

    global_weights = global_model.get_weights()

    # ─────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────

    tracker = MetricsTracker()

    writer = get_writer(args)

    log_dir = os.path.join(
        args.save_dir,
        "logs"
    )

    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(
        log_dir,
        f"{args.exp_name}.log"
    )

    start_time = time.time()

    # ─────────────────────────────────────────
    # FL rounds
    # ─────────────────────────────────────────

    round_bar = tqdm(range(1, args.rounds + 1), desc='Rounds', unit='rnd', ncols=80)

    for rnd in round_bar:

        global_model.set_weights(global_weights)

        # select clients
        n_select = max(
            1,
            int(args.frac * args.num_clients)
        )

        selected = np.random.choice(
            args.num_clients,
            n_select,
            replace=False
        ).tolist()

        # simulate dropout
        active = apply_dropout(
            selected,
            args.dropout
        )

        n_active = len(active)

        local_weights_list = []

        local_losses = []

        local_sizes = []

        # ─────────────────────────────────────
        # Local training
        # ─────────────────────────────────────

        for cid in tqdm(active, desc=f'  Round {rnd:3d} clients', unit='client', ncols=80, leave=False):

            idxs = client_idxs[cid]

            if len(idxs) == 0:
                continue

            if args.algorithm == "fedprox":

                trainer = LocalUpdateProx(
                    args,
                    train_dataset,
                    idxs,
                )

                w, loss = trainer.train(
                    global_model,
                    global_model
                )

            else:

                trainer = LocalUpdate(
                    args,
                    train_dataset,
                    idxs,
                )

                w, loss = trainer.train(
                    global_model
                )

            local_weights_list.append(w)

            local_losses.append(loss)

            local_sizes.append(len(idxs))

        # skip if nobody returned
        if not local_weights_list:

            print(
                f"[Round {rnd:3d}] "
                f"No clients returned."
            )

            continue

        # ─────────────────────────────────────
        # Aggregation
        # ─────────────────────────────────────

        global_weights = fedavg_aggregate(
            local_weights_list,
            local_sizes
        )

        global_model.set_weights(
            global_weights
        )

        # ─────────────────────────────────────
        # Evaluation
        # ─────────────────────────────────────

        if rnd % args.log_interval == 0 \
           or rnd == args.rounds:

            test_acc, test_loss = evaluate_global(
                global_model,
                test_dataset
            )

            train_loss = weighted_average(
                local_losses,
                local_sizes
            )

            # client drift
            drift = compute_client_drift(
                local_weights_list,
                global_weights
            )

            # fairness
            per_acc = evaluate_per_client(
                global_model,
                train_dataset,
                client_idxs,
                active,
            )

            fairness = compute_fairness(
                per_acc
            )

            tracker.update(
                rnd,
                train_loss,
                test_loss,
                test_acc,
                drift,
                fairness,
                n_active,
            )

            # ─────────────────────────────────
            # TensorBoard
            # ─────────────────────────────────

            if writer:

                with writer.as_default():

                    tf.summary.scalar(
                        "Loss/train",
                        train_loss,
                        step=rnd
                    )

                    tf.summary.scalar(
                        "Loss/test",
                        test_loss,
                        step=rnd
                    )

                    tf.summary.scalar(
                        "Accuracy/test",
                        test_acc,
                        step=rnd
                    )

                    tf.summary.scalar(
                        "ClientDrift",
                        drift,
                        step=rnd
                    )

                    tf.summary.scalar(
                        "Fairness",
                        fairness,
                        step=rnd
                    )

                    tf.summary.scalar(
                        "ActiveClients",
                        n_active,
                        step=rnd
                    )

            elapsed = time.time() - start_time

            round_bar.set_postfix(
                acc=f"{test_acc*100:.1f}%",
                loss=f"{test_loss:.3f}",
                drift=f"{drift:.3f}",
            )

            log_line = (
                f"[Round {rnd:3d}/{args.rounds}] "
                f"TrainLoss: {train_loss:.4f}  "
                f"TestLoss: {test_loss:.4f}  "
                f"TestAcc: {test_acc*100:.2f}%  "
                f"Drift: {drift:.4f}  "
                f"Fairness: {fairness:.4f}  "
                f"Active: {n_active}/{n_select}  "
                f"Time: {elapsed:.1f}s"
            )

            if args.verbose:
                print(log_line)

            with open(log_path, "a") as f:
                f.write(log_line + "\n")

    # ─────────────────────────────────────────
    # Save results
    # ─────────────────────────────────────────

    results_dir = os.path.join(args.save_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(
    results_dir,
    f"{args.exp_name}.json"
    )

    summary = tracker.summary()

    def to_json_safe(obj):
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]

        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)

        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)

        return obj
    output = {
        "config": vars(args),
        "summary": summary,
        "history": tracker.to_dict(),
    }

    output = to_json_safe(output)

    with open(results_path, "w") as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print(f"\n✓ Experiment complete.")

    print(
        f"Best accuracy : "
        f"{summary.get('best_acc', 0)*100:.2f}%"
    )

    print(
        f"Final accuracy: "
        f"{summary.get('final_acc', 0)*100:.2f}%"
    )

    print(
        f"Mean drift    : "
        f"{summary.get('mean_drift', 0):.4f}"
    )

    print(
        f"Results saved : "
        f"{results_path}"
    )

    print(
        f"Log saved     : "
        f"{log_path}\n"
    )

    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    args = args_parser()

    run_experiment(args)