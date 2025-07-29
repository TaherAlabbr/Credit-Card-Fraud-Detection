import pickle
import sys

from cli_args import get_args
from credit_fraud_utils_data import load_data
from credit_fraud_utils_eval import evaluate


def load_model(model_path: str = "model.pkl"):
    """
    Load a trained model or pipeline from a .pkl file.

    Parameters:
    -----------
    model_path : str
        Path to the saved pickle file.

    Returns:
    --------
    object or None
        The deserialized model or pipeline if successful, else None.
    """
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"[INFO] Model loaded successfully from '{model_path}'")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Model file not found at: '{model_path}'", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to load model from '{model_path}': {e}", file=sys.stderr)
    return None


def main():
    args = get_args()

    # Load the data
    X_train, X_test, y_train, y_test = load_data(args.train_dir, args.test_dir)

    # Load the model bundle
    model_bundle = load_model(args.load_dir)
    if model_bundle is None:
        print("[ERROR] Model loading failed. Exiting.", file=sys.stderr)
        sys.exit(1)

    model = model_bundle.get('model', None)
    if model is None:
        print("[ERROR] No 'model' key found in loaded bundle.", file=sys.stderr)
        sys.exit(1)

    # Evaluate on training and test sets
    evaluate(model, X_train, y_train, split='Training')
    evaluate(model, X_test, y_test, split='Testing')


if __name__ == '__main__':
    main()
