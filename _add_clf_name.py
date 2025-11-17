import os
import joblib
from sklearn.multiclass import OneVsRestClassifier

# Directory where joblib files live
MODEL_DIR = "Classifier"

# Mapping from relative path -> friendly classifier name
MODEL_NAMES = {
    "semeval_test/semeval_task_a.joblib": "semeval_2020_task_a",
    "semeval_test/semeval_task_b.joblib": "semeval_2020_task_b",
    "semeval_test/semeval_task_c.joblib": "semeval_2020_task_c",
    "jigsaw/jigsaw.joblib": "jigsaw",
    "goemotion/goemotions.joblib": "goemotions",
    "davidson_etal/davidson.joblib": "davidson"
}

def detect_multilabel(clf):
    """Detect whether a classifier is multi-label."""
    return isinstance(clf, OneVsRestClassifier)


def load_or_fix_model(path, expected_name):
    print(f"Loading → {path}")
    obj = joblib.load(path)

    # Fix missing classifier_name
    if "classifier_name" not in obj:
        print(f"  → Adding classifier_name = {expected_name}")
        obj["classifier_name"] = expected_name

    # Fix missing multi_label using real detection
    clf = obj.get("classifier")
    if clf is None:
        print("  ⚠ No classifier found in joblib!")
    else:
        detected = detect_multilabel(clf)
        if "multi_label" not in obj or obj["multi_label"] != detected:
            print(f"  → Setting multi_label = {detected}")
            obj["multi_label"] = detected

    # Validate label_names
    if "label_names" not in obj:
        print("  ⚠ Warning: label_names missing!")
    else:
        print(f"  → label_names found ({len(obj['label_names'])} labels)")

    # Save updated object
    joblib.dump(obj, path)

    return obj


def load_all_classifiers():
    loaded = {}
    for filename, expected_name in MODEL_NAMES.items():
        path = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(path):
            print(f"⚠ File not found → {path}")
            continue

        loaded[expected_name] = load_or_fix_model(path, expected_name)

    return loaded


if __name__ == "__main__":
    models = load_all_classifiers()

    print("\n=== Loaded Models ===")
    for name, obj in models.items():
        print(f"{name}: labels={len(obj['label_names'])}, "
              f"multi_label={obj['multi_label']}, "
              f"classifier_name={obj['classifier_name']}")
