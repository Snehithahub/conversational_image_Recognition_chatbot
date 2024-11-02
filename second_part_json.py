import nltk
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Download required nltk resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
stemmer = WordNetLemmatizer()

# Load intents JSON file
with open('/content/intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?']

# Process each intent pattern 
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words in each pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lower, and sort words
words = [stemmer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"{len(documents)} documents: {documents}")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [stemmer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

# Synonym replacement for data augmentation
def synonym_replacement(tokens, limit):
    augmented_sentences = []
    for i in range(len(tokens)):
        synonyms = []
        for syn in wordnet.synsets(tokens[i]):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        if synonyms:
            num_augmentations = min(limit, len(synonyms))
            sampled_synonyms = random.sample(synonyms, num_augmentations)
            for synonym in sampled_synonyms:
                augmented_tokens = tokens[:i] + [synonym] + tokens[i + 1:]
                augmented_sentences.append(' '.join(augmented_tokens))
    return augmented_sentences

# Augment training data
augmented_data = []
limit_per_tag = 100

for bag, output_row in training:
    tokens = [words[j] for j in range(len(words)) if bag[j] == 1]
    augmented_sentences = synonym_replacement(tokens, limit_per_tag)
    for augmented_sentence in augmented_sentences:
        augmented_bag = [1 if word in augmented_sentence.split() else 0 for word in words]
        augmented_data.append([augmented_bag, output_row])

# Convert both lists to arrays with the same structure
training = np.array(training, dtype=object)
augmented_data = np.array(augmented_data, dtype=object)

# Concatenate training and augmented data if shapes are consistent
combined_data = np.concatenate((training, augmented_data), axis=0)
random.shuffle(combined_data)

# Define the CustomDataset class
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Create DataLoader for training and testing data
batch_size = 64
train_x = torch.tensor([data[0] for data in combined_data]).float()
train_y = torch.tensor([data[1] for data in combined_data]).float()
train_loader = DataLoader(CustomDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

# The rest of your model code goes here...
