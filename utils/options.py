import argparse


def args_parser():
    parser = argparse.ArgumentParser(description="Federated Learning - FedAvg / FedProx")

    # ── Federated settings ──────────────────────────────────────────────────
    parser.add_argument("--algorithm", type=str, default="fedavg",
                        choices=["fedavg", "fedprox"],
                        help="Aggregation algorithm (fedavg | fedprox)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of communication rounds")
    parser.add_argument("--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument("--frac", type=float, default=0.25,
                        help="Fraction of clients selected per round")
    parser.add_argument("--local_epochs", type=int, default=2,
                        help="Local training epochs per round")
    parser.add_argument("--local_bs", type=int, default=32,
                        help="Local mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Local learning rate")

    # ── Non-IID / Dirichlet ─────────────────────────────────────────────────
    parser.add_argument("--iid", action="store_true",
                        help="Use IID data distribution (overrides --alpha)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha for Non-IID (smaller → more skewed)")

    # ── Client dropout ──────────────────────────────────────────────────────
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Client dropout probability per round [0, 1)")

    # ── FedProx ─────────────────────────────────────────────────────────────
    parser.add_argument("--mu", type=float, default=0.01,
                        help="Proximal term coefficient μ for FedProx")

    # ── Model / Dataset ─────────────────────────────────────────────────────
    parser.add_argument("--model", type=str, default="cnn",
                        choices=["cnn", "mlp"],
                        help="Neural network model (cnn | mlp)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar"],
                        help="Dataset (mnist | cifar)")

    # ── Reproducibility ─────────────────────────────────────────────────────
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # ── Logging / Output ─────────────────────────────────────────────────────
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log metrics every N rounds")
    parser.add_argument("--save_dir", type=str, default="experiments",
                        help="Root directory to save logs and results")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name tag (auto-generated if empty)")
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-round metrics to console")

    args = parser.parse_args()

    # Auto-generate experiment name if not provided
    if not args.exp_name:
        iid_tag = "iid" if args.iid else f"alpha{args.alpha}"
        args.exp_name = (
            f"{args.algorithm}_{args.dataset}_{args.model}"
            f"_{iid_tag}_drop{int(args.dropout * 100)}"
            f"_seed{args.seed}"
        )

    return args