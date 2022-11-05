from transformers import pipeline

from transformers import AutoTokenizer, AutoModelForTokenClassification

from datasets import load_dataset

imdb = load_dataset("imdb")

# use only part of dataset
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

# preprocess data using AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

