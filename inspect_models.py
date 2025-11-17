import joblib
import pprint
import sys
import os
import joblib

# Directory where joblib files live
MODEL_DIR = "/Classifier"

# Mapping from relative path -> friendly classifier name
MODEL_NAMES = {
    "semeval_test/semeval_task_a.joblib": "semeval_2020_task_a",
    "semeval_test/semeval_task_b.joblib": "semeval_2020_task_b",
    "semeval_test/semeval_task_c.joblib": "semeval_2020_task_c",
    "jigsaw/jigsaw.joblib": "jigsaw",
    "goemotion/goemotions.joblib": "goemotions",
    "davidson_etal/davidson.joblib": "davidson"
}

def inspect_joblib(path):
    print(f"\n=== Inspecting Joblib File ===")
    print(f"Path: {path}\n")

    obj = joblib.load(path)

    print("=== Top-level Keys ===")
    for key in obj.keys():
        print(f" - {key}")

    print("\n=== Full Contents ===")
    pp = pprint.PrettyPrinter(indent=2, width=120)
    pp.pprint(obj)


if __name__ == "__main__":
    #build paths 
    for filename in MODEL_NAMES.keys():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            inspect_joblib(path=path)
