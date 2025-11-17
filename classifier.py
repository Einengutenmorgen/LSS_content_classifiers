#classifier.py

import os
import json
import logging
from dotenv import load_dotenv

import pydantic
import requests
import typing
from typing import Optional
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

load_dotenv()


import os
import time
import typing
import requests
import pydantic

class LLM(pydantic.BaseModel):
    
    api_key: str = os.getenv("HF_API_KEY")
    url: str = "https://router.huggingface.co/hf-inference/models/mixedbread-ai/mxbai-embed-large-v1/pipeline/feature-extraction"

    if not api_key:
        raise ValueError("HF_API_KEY environment variable not set.")

    def _query(self, payload, max_retries=6):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                # Check HTTP error
                if not response.ok:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )

                # Attempt JSON decode
                try:
                    return response.json()
                except Exception:
                    print("Non-JSON response from HF endpoint:")
                    print(response.text[:500])
                    raise RuntimeError("Received non-JSON response")

            except Exception as e:
                print(f"⚠ Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 + attempt * 2)

    def extract(self, text, batch_size: int = 450):
        """
        Returns embeddings for a string OR list of strings.
        Automatically batches lists.
        """

        # Single string → send directly
        if isinstance(text, str):
            return self._query({"inputs": text})

        # Case 2: List of strings → batch
        if isinstance(text, list):
            result = []
            for i in range(0, len(text), batch_size):
                batch = text[i:i+batch_size]
                print(f"Embedding batch {i}–{i+len(batch)}...")
                emb = self._query({"inputs": batch})
                result.extend(emb)
            return result

        raise TypeError("text must be a string or list of strings")



def train_classifier(
    text: list[str],
    labels,
    report: bool,
    emb_path: Optional[str] = None,
    classifier_name: Optional[str] = None
) -> dict:

    # --- Load or compute embeddings ---
    if emb_path and os.path.exists(emb_path):
        print(f"Loading existing embeddings from {emb_path}")
        embeddings = np.load(emb_path)
        if embeddings.shape[0] != len(text):
            raise ValueError("Embedding count does not match text count.")
    else:
        print("Computing new embeddings...")
        embeddings = LLM().extract(text)
        embeddings = np.array(embeddings)

        if emb_path:
            np.save(emb_path, embeddings)
            print(f"Saved embeddings to {emb_path}")

    # --- Determine label type ---
    if isinstance(labels[0], dict):
        # MULTI-label
        label_names = list(labels[0].keys())
        y = np.array([[sample[k] for k in label_names] for sample in labels])
        multi_label = True
    else:
        # SINGLE-label
        le = LabelEncoder()
        y = le.fit_transform(labels)
        label_names = list(le.classes_)
        multi_label = False

    # --- Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42
    )

    # --- Build model ---
    if multi_label:
        clf = OneVsRestClassifier(
            MLPClassifier(hidden_layer_sizes=(300,), max_iter=300, random_state=42)
        )
    else:
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)

    # --- Train ---
    clf.fit(X_train, y_train)

    # --- Build return dict ---
    result = {
        "classifier_name": classifier_name,
        "classifier": clf,
        "label_names": label_names,
        "multi_label": multi_label
    }

    # --- Metrics ---
    if report:
        if multi_label:
            y_pred = (clf.predict_proba(X_test) >= 0.5).astype(int)
            result["metrics"] = {
                "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
                "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "classification_report": classification_report(
                    y_test, y_pred, target_names=label_names, zero_division=0
                )
            }
        else:
            y_pred = clf.predict(X_test)
            result["metrics"] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
                "classification_report": classification_report(
                    y_test, y_pred, target_names=label_names, zero_division=0
                )
            }

    return result






if __name__ == "__main__":
    texts = [
        "I love programming in Python!",
        "The weather is great today.",
        "I hate getting up early.",
        "This new movie is fantastic!",
        "I'm so tired of all this rain.",
        "What a sunny day!"
    ]
    labels = [
        "positive",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive"
    ]

    result = train_classifier(texts, labels, report=False, emb_path="test_embeddings_sing.npy", classifier_name="sentiment_classifier")



    labels_num = [
        1,
        0,
        1,
        2,
        2,
        0
    ]

    result = train_classifier(texts, labels_num, report=False, emb_path="test_embeddings_num.npy", classifier_name="numeric_classifier")

    labels_multi = [
        {"love":0, "hate": 1, "fear": 0, "joy":1, "sadness":1},
        {"love":0, "hate": 0, "fear": 0, "joy":1, "sadness":0},
        {"love":0, "hate": 1, "fear": 0, "joy":0, "sadness":1},
        {"love":1, "hate": 0, "fear": 0, "joy":1, "sadness":0},
        {"love":0, "hate": 0, "fear": 0, "joy":0, "sadness":1},
        {"love":0, "hate": 0, "fear": 0, "joy":1, "sadness":0}
    ]
    result_multi = train_classifier(texts, labels_multi, report=False, emb_path="test_embeddings_multi.npy", classifier_name="multi_emotion_classifier")
