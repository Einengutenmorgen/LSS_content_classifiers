# **README – Klassifier Übersicht & Nutzung**

Dieses Projekt enthält mehrere trainierte Textklassifikationsmodelle.  
Alle Modelle liegen als **Joblib-Dictionaries** mit konsistenter Struktur vor:

```python
{
  "classifier_name": str,
  "classifier": sklearn-Modell,
  "label_names": list[str],
  "multi_label": bool,
  "metrics": dict
}
````

Damit sind alle Modelle ohne Anpassungen direkt inferenzfähig.

---

## **1. Verwendete Klassifier**

### **SemEval 2020 Tasks (Offensive Language Identification)**

| Modell                | Labels        | Typ          |
| --------------------- | ------------- | ------------ |
| `semeval_2020_task_a` | NOT, OFF      | Single-Label |
| `semeval_2020_task_b` | TIN, UNT      | Single-Label |
| `semeval_2020_task_c` | GRP, IND, OTH | Single-Label |

---

### **Jigsaw Toxicity (6 Klassen)**

**Typ:** Multi-Label (OneVsRest)

**Labels:**
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

### **GoEmotions (28 Klassen)**

**Typ:** Multi-Label (OneVsRest)

**Labels:**
27 Emotionen + *neutral*
(Hinweis: neutral kommt im Trainingsset kaum vor)

---

### **Davidson Hate Speech (3 Klassen)**

**Typ:** Single-Label

**Labels:**
`0` (hate), `1` (offensive), `2` (neutral)

---

## **2. Performance (Kurzüberblick)**

### **SemEval**

* Task A Accuracy: **0.90**
* Task B Accuracy: **0.74**
* Task C Accuracy: **0.82**

### **Jigsaw**

* F1-Macro: **0.55**
* F1-Micro: **0.67**

### **GoEmotions**

* F1-Macro: **0.25**
* F1-Micro: **0.49**

### **Davidson**

* Accuracy: **0.86**

---

## **3. Modelle laden**

```python
import joblib

obj = joblib.load("path/to/model.joblib")
clf = obj["classifier"]
labels = obj["label_names"]
is_multi = obj["multi_label"]
```

---

## **4. Inferenz**

### **Embeddings erzeugen**

```python
from classifier import LLM
import os

emb = LLM(api_key=os.getenv("HF_API_KEY")).extract("Text here")
```

---

### **Single-Label Prediction**

```python
pred_idx = clf.predict([emb])[0]
label = labels[pred_idx]
```

---

### **Multi-Label Prediction**

```python
vec = clf.predict([emb])[0]
result = {label: int(v) for label, v in zip(labels, vec)}
```

---

## **5. Modell-Inspektion**

Joblib-Inhalte anzeigen:

```bash
python inspect_models.py
```

Gibt u. a. folgende Informationen aus:

* classifier_name
* label_names
* metrics
* multi_label
* sklearn Modellstruktur

---

## **6. Speicherorte der Modelle**

```
semeval_test/semeval_task_a.joblib
semeval_test/semeval_task_b.joblib
semeval_test/semeval_task_c.joblib
jigsaw/jigsaw.joblib
goemotion/goemotions.joblib
davidson_etal/davidson.joblib
```

Alle Modelle enthalten korrekt:

* `"classifier_name"`
* `"multi_label"`
* `"label_names"`
* `"metrics"`

---

## **Fertig**

Alle Klassifier sind vereinheitlicht, bereinigt und direkt einsatzbereit.
Single-Label und Multi-Label Inferenz funktioniert nun mit einer identischen API.

```
```
