"""
This file converts to the following format:
John NOUN
loves VERB
Mary NOUN

Bobby NOUN
likes VERB
to PRP
run VERB
"""
from datasets import load_dataset

dataset = load_dataset("universal_dependencies", "en_gum")



