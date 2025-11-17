# run_training.py

from classifier import train_classifier
import ast 
import pandas as pd
import joblib

# --- TASK A ---
# print("Training Task A...")
# #add column name
# df_a_labels = pd.read_csv('semeval_test/test_a_labels.csv', header=None, names=['id', 'label'])
# df_a_tweets = pd.read_csv('semeval_test/test_a_tweets.tsv', sep='\t')

# # combine based on column id 
# df_a = pd.merge(df_a_tweets, df_a_labels, on='id')
# texts = df_a['tweet'].tolist()
# labels = df_a['label'].tolist()

# classifier_a = train_classifier(texts, labels, report=True)
# joblib.dump(classifier_a, 'semeval_test/semeval_task_a.joblib')


# --- TASK B ---
# print("Training Task B...")
# df_b_labels = pd.read_csv('semeval_test/test_b_labels.csv', header=None, names=['id', 'label'])
# df_b_tweets = pd.read_csv('semeval_test/test_b_tweets.tsv', sep='\t')

# df_b = pd.merge(df_b_tweets, df_b_labels, on='id')
# texts = df_b['tweet'].tolist()
# labels = df_b['label'].tolist()

# classifier_b = train_classifier(texts, labels, report=True)

# joblib.dump(classifier_b, 'semeval_test/semeval_task_b.joblib')


# --- TASK C ---
# print("Training Task C...")
# df_c_labels = pd.read_csv('semeval_test/test_c_labels.csv', header=None, names=['id', 'label'])
# df_c_tweets = pd.read_csv('semeval_test/test_c_tweets.tsv', sep='\t')

# df_c = pd.merge(df_c_tweets, df_c_labels, on='id')
# texts = df_c['tweet'].tolist()
# labels = df_c['label'].tolist()

# classifier_c = train_classifier(texts, labels, report=True)
# joblib.dump(classifier_c, 'semeval_test/semeval_task_c.joblib')

# print("All tasks completed successfully.")

# --- Task D ---
print("Training Task D...")
df_jigsaw=pd.read_csv('/Classifier/jigsaw/jigsaw_train.csv')
df_jigsaw["labels_dict"] = df_jigsaw["labels_dict"].apply(ast.literal_eval)
df_jigsaw=df_jigsaw[:25000]
df_jigsaw_texts=df_jigsaw['comment_text'].tolist()
df_jigsaw_labels=df_jigsaw['labels_dict'].tolist()

classifier_d = train_classifier(df_jigsaw_texts, df_jigsaw_labels, report=True, emb_path='/Classifier/jigsaw/embeddings_jigsaw.npy', classifier_name='jigsaw_classifier')
joblib.dump(classifier_d, '/Classifier/jigsaw/jigsaw.joblib')

print("All tasks completed successfully.")

# # --- Task E ---
print("Training Task E...")

df_goemotions=pd.read_csv('/Classifier/goemotion/goe_f_b_213k.csv.csv')
df_goemotions["labels_dict"] = df_goemotions["labels_dict"].apply(ast.literal_eval)
df_goemotions=df_goemotions[:75000]
df_goemotions_texts=df_goemotions['text'].tolist()
df_goemotions_labels=df_goemotions['labels_dict'].tolist()
classifier_e = train_classifier(df_goemotions_texts, df_goemotions_labels, report=True, emb_path='/Classifier/goemotion/embeddings_goemotion.npy', classifier_name='goemotion_classifier')
joblib.dump(classifier_e, '/Classifier/goemotion/goemotions.joblib')

print("All tasks completed successfully.")

# --- Task F ---
# print("Training Task F...")

# df_davidson=pd.read_csv('/Classifier/davidson_etal/labeled_data.csv')
# df_davidson_texts=df_davidson['tweet'].tolist()
# df_davidson_labels=df_davidson['class'].tolist()
# classifier_f = train_classifier(df_davidson_texts, df_davidson_labels,  report=True, emb_path='/Classifier/davidson_etal/embeddings_davidson.npy')
# joblib.dump(classifier_f, '/Classifier/davidson_etal/davidson.joblib')