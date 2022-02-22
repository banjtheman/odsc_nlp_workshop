from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import pandas as pd
from datasets import Dataset
from datasets import load_dataset

# Start with base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# State number of labels
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Load training data
test_data = pd.read_csv("helpful_sentences_test_num.csv")
train_data = pd.read_csv("helpful_sentences_train_num.csv")

# Convert to Dataset format
test_dataset = Dataset.from_pandas(test_data)
train_dataset = Dataset.from_pandas(train_data)

# Transform to tokens
tokenized_test = test_dataset.map(preprocess_function, batched=True)
tokenized_train = train_dataset.map(preprocess_function, batched=True)

# Can eval on accuracy
from datasets import load_metric
import numpy as np
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Set the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)

# Start the training, it will save results every 500 steps
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# To resume training replace xxx with last saved folder
# trainer.train("results/checkpoint-xxx")

# Evaluate the model
trainer.evaluate()
