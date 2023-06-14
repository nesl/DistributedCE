from transformers import pipeline
from datasets import Dataset, Image
import pandas as pd
import PIL
from evaluate import evaluator
import argparse

#Test classifier

parser = argparse.ArgumentParser(description='Testing image classifier')

parser.add_argument('--metadata', type=str, help='Metadata file')
parser.add_argument('--model', type=str, help='Model directory')
parser.add_argument('--eval', action="store_true", help = "Get aggregated evaluation metrics")
parser.add_argument('--image', type=str, default='', help='Image file to test')

args = parser.parse_args()


if not args.image:

	new_pd = pd.read_csv(args.metadata)
	#print(new_pd["image"][0])

	ds = Dataset.from_pandas(new_pd, preserve_index=False).cast_column("image", Image())


	if args.eval:

		task_evaluator = evaluator("image-classification")

		results = task_evaluator.compute(model_or_pipeline=args.model, data=ds, label_column="label", metric="accuracy", strategy="bootstrap", label_mapping={'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2, 'LABEL_5': 5})

		print(results)

	else:

		image = ds["image"]


		classifier = pipeline("image-classification", model=args.model, device=0)

		results = classifier(image)

		for r_idx,r in enumerate(results):
			print(r, " " + str(ds["label"][r_idx]))
			
else:
	

	image = PIL.Image.open(args.image)

	classifier = pipeline("image-classification", model=args.model)

	results = classifier(image)
	
	print(results)
