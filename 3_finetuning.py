#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning

# In[1]:


import pandas as pd
import pyspark
import sklearn
import torch
import transformers

from IPython.display import display
from pyspark.sql.functions import *
from pyspark.sql.types import *


# In[2]:


spark = (
    pyspark.sql.SparkSession.builder
    .master("local[*]")
    .appName("bod-seats")
    .config("spark.driver.memory", "16g")
    .getOrCreate()
)
spark


# In[3]:


train = spark.read.format("parquet").load("train.parquet")
train = train.withColumn("label", col("label").astype(BooleanType()))
train = train.withColumn("text", col("text").dropFields("length", "offset_mapping", "special_tokens_mask"))
train = train.withColumn("text", col("text").withField("labels", col("label").astype(ByteType()))).drop("label")
train.printSchema()
train = train.toPandas()
test = spark.read.format("parquet").load("test.parquet")
test = test.withColumn("label", col("label").astype(BooleanType()))
test = test.withColumn("text", col("text").dropFields("length", "offset_mapping", "special_tokens_mask"))
test = test.withColumn("text", col("text").withField("labels", col("label").astype(ByteType()))).drop("label")
test.printSchema()
test = test.toPandas()


# In[4]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.df = df

    def __getitem__(self, index) -> dict[str, list[int]]:
        return self.df.loc[index, "text"].asDict()

    def __len__(self) -> int:
        return self.df.shape[0]

train_dataset = Dataset(train)
test_dataset = Dataset(test)


# In[5]:


args = transformers.TrainingArguments(
    output_dir="3_finetuning/output",
    num_train_epochs=8,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_dir="3_finetuning/logging",
    logging_steps=64,
    dataloader_num_workers=64,
    evaluation_strategy="steps",
    eval_steps=64,
    save_steps=64,
    fp16=True,
    fp16_opt_level="O3",
    learning_rate=5e-5,
)

def compute_metrics(output):
    labels = output.label_ids
    index = labels != -100
    labels = labels[index]
    predictions = output.predictions.argmax(-1)[index]
    metrics = {}
    for average in ("micro", "macro", "weighted"):
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
            labels, predictions, average=average)
        metrics[f"{average}_precision"] = precision
        metrics[f"{average}_recall"] = recall
        metrics[f"{average}_f1"] = f1
    metrics["accuracy"] = sklearn.metrics.accuracy_score(labels, predictions)
    return metrics

model = transformers.AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=2)
tokenizer = transformers.AlbertTokenizerFast.from_pretrained("albert-base-v2")
data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=32)
trainer = transformers.Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)


# In[ ]:


trainer.train()

