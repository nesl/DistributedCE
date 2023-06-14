import pandas as pd
from datasets import Dataset, Image
import pdb
from transformers import AutoImageProcessor, DefaultDataCollator, pipeline, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import evaluate
import numpy as np


#Train classifier

df_train = pd.read_csv("crop_training/metadata.csv")
df_val = pd.read_csv("crop_val/metadata.csv")
df_base = pd.read_csv("crop_base/metadata.csv")
df_onroad = pd.read_csv("crop_onroad/metadata.csv")
df_onroad = pd.read_csv("crop_temp/metadata.csv")

new_pd = pd.concat([df_train,df_val,df_base,df_onroad])

labels_of_interest = [0,1,2,5]


number_per_class = min([sum(new_pd["label"] == label) for label in labels_of_interest]) #Get class with minimum amount of samples


data_indices = []

for label in labels_of_interest:
	data_indices.extend(new_pd.iloc[np.where(new_pd["label"] == label)[0]].sample(number_per_class).index.values.tolist())

ds = Dataset.from_pandas(new_pd.iloc[data_indices], preserve_index=False).cast_column("image", Image())


ds = ds.train_test_split(test_size=0.2)


checkpoint = "google/vit-base-patch16-224-in21k"

image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

resize = Resize((224,224))

_transforms = Compose([ToTensor(), resize, normalize])

def transforms(examples):

	examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]

	del examples["image"]

	return examples
	

ds = ds.with_transform(transforms)

data_collator = DefaultDataCollator()

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):

	predictions, labels = eval_pred

	predictions = np.argmax(predictions, axis=1)

	return accuracy.compute(predictions=predictions, references=labels)
	
	

#labels = ds["train"].features["label"].names
	
model = AutoModelForImageClassification.from_pretrained(checkpoint,num_labels=max(labels_of_interest)+1)


training_args = TrainingArguments(output_dir="img_classifier", remove_unused_columns=False, evaluation_strategy="epoch", save_strategy="epoch", learning_rate=5e-5, per_device_train_batch_size=16, gradient_accumulation_steps=4, per_device_eval_batch_size=16, num_train_epochs=3, warmup_ratio=0.1, logging_steps=10, load_best_model_at_end=True, metric_for_best_model="accuracy", push_to_hub=False)

trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=ds["train"], eval_dataset=ds["test"], tokenizer=image_processor, compute_metrics=compute_metrics)



trainer.train()


