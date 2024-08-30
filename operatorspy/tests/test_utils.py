def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU test",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run CUDA test",
    )
    parser.add_argument(
        "--bang",
        action="store_true",
        help="Run BANG test",
    )
    parser.add_argument(
        "--musa",
        action="store_true",
        help="Run MUSA test",
    )

    return parser.parse_args()
