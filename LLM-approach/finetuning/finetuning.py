#########################################
# Code to finetune a model (high-level) #
# Using HuggingFace tutorials           #
#########################################

import pandas as pd
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Configuration
MODEL_NAME = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
train_csv = '/home/vellard/malis/clustering-no-split/split/clusters_train.csv'
val_csv = '/home/vellard/malis/clustering-no-split/split/clusters_val.csv'
output_dir = './fine_tuned_model'
batch_size = 5
epochs = 7
learning_rate = 2e-5
warmup_steps = 100

train_df = pd.read_csv(train_csv, low_memory=False)
val_df = pd.read_csv(val_csv, low_memory=False)

# Preprocessing of clusters: map the 158 clusters to sequential labels (from 0->157)
# to avoid non-existing clusters labels (that could have been removed during the cleaning of clusters)
avoid non-existing clusters labels
train_df['Cluster ID'] = train_df['Cluster ID'].astype(int)
val_df['Cluster ID'] = val_df['Cluster ID'].astype(int)

unique_train_labels = sorted(train_df['Cluster ID'].unique())
label_mapping = {orig_label: new_label for new_label, orig_label in enumerate(unique_train_labels)}
train_df['Mapped Label'] = train_df['Cluster ID'].map(label_mapping)
val_df = val_df[val_df['Cluster ID'].isin(label_mapping.keys())].copy()#Filter validation clusters to existing training clusters
val_df['Mapped Label'] = val_df['Cluster ID'].map(label_mapping)

num_labels = len(label_mapping)
print(f"Number of labels: {num_labels}")#To verify

#HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['Playlist Title', 'Mapped Label']])
val_dataset = Dataset.from_pandas(val_df[['Playlist Title', 'Mapped Label']])

#--------------------------------------#
#Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

def tokenize_function(examples):#because of errors in tokenization
    texts = [str(text) for text in examples["Playlist Title"]]
    return tokenizer(texts, truncation=True, padding="max_length")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

#use good format for PyTorch
tokenized_train = tokenized_train.rename_column("Mapped Label", "labels")
tokenized_val = tokenized_val.rename_column("Mapped Label", "labels")
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#Training arguments
#Ensure to have matching strategies for evaluation and saving ("epoch")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_steps=warmup_steps,
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#----------------------------------#
# Train

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(output_dir)
print(f"Model saved to {output_dir}")
