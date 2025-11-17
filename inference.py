import joblib
from classifier import LLM
import os

# Load objects
#semeval
obj_a = joblib.load("/Classifier/semeval_test/semeval_task_a.joblib")
obj_b = joblib.load("/Classifier/semeval_test/semeval_task_b.joblib")
obj_c = joblib.load("/Classifier/semeval_test/semeval_task_c.joblib")
#goemotion
obj_g=joblib.load("/Classifier/goemotion/goemotions.joblib")
print("loaded goemotion joblib", obj_g)
#jigsaw
obj_j=joblib.load("/Classifier/jigsaw/jigsaw.joblib")
#davidson
obj_f=joblib.load("/Classifier/davidson_etal/davidson.joblib")

clf_a = obj_a["classifier"]
clf_b = obj_b["classifier"]
clf_c = obj_c["classifier"]

clf_g = obj_g["classifier"]

clf_j = obj_j["classifier"]

clf_f = obj_f["classifier"]

labels_a = obj_a["label_names"]
print("Labels A:", labels_a, len(labels_a))
labels_b = obj_b["label_names"]
print("Labels B:", labels_b, len(labels_b))
labels_c = obj_c["label_names"]
print("Labels C:", labels_c, len(labels_c))
labels_g = obj_g["label_names"]
print("Labels GoEmotions:", labels_g, len(labels_g))
labels_j = obj_j["label_names"]
print("Labels Jigsaw:", labels_j, len(labels_j))
labels_f = obj_f["label_names"]

def decode_multilabel(label_names, outputs):
    return {name: int(o) for name, o in zip(label_names, outputs)}

def predict_all(text):
    embeddings = LLM(api_key=os.getenv('HF_API_KEY')).extract(text)

    a_idx = clf_a.predict([embeddings])[0]
    b_idx = clf_b.predict([embeddings])[0]
    c_idx = clf_c.predict([embeddings])[0]

    # multi-label prediction vectors
    g_vec = clf_g.predict([embeddings])[0]
    j_vec = clf_j.predict([embeddings])[0]

    f_idx = clf_f.predict([embeddings])[0]

    return {
        "task_a": labels_a[a_idx],
        "task_b": labels_b[b_idx],
        "task_c": labels_c[c_idx],

        # multi-label: decode vector → dict
        "goemotions": decode_multilabel(labels_g, g_vec),
        "jigsaw": decode_multilabel(labels_j, j_vec),

        "davidson": labels_f[f_idx]
    }


if __name__ == "__main__":
    example = "This is a sample tweet."
    preds = predict_all(example)
    print("Input:", example)
    print("Task A:", preds["task_a"])
    print("Task B:", preds["task_b"])
    print("Task C:", preds["task_c"])
    print("GoEmotions:", preds["goemotions"])
    print("Jigsaw:", preds["jigsaw"])
    print("Davidson:", preds["davidson"])
    print("GoEmotions:", preds["goemotions"])
