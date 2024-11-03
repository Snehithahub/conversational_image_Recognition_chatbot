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

# Concatenate training and augmented data
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

# Create DataLoader for training data
batch_size = 64
train_x = torch.tensor([data[0] for data in combined_data]).float()
train_y = torch.tensor([data[1] for data in combined_data]).float()
train_loader = DataLoader(CustomDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

# Initialize model, loss function, and optimizer
input_size = len(train_x[0])
hidden_size = 8
output_size = len(train_y[0])
model = NeuralNetwork(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, targets) * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Load model function
def load_model(model_path, input_size, hidden_size, output_size):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to preprocess the input sentence
def preprocess_sentence(sentence, words):
    sentence_words = sentence.lower().split()
    sentence_words = [word for word in sentence_words if word in words]
    return sentence_words

# Function to convert the preprocessed sentence into a feature vector
def sentence_to_features(sentence_words, words):
    features = [1 if word in sentence_words else 0 for word in words]
    return torch.tensor(features).float().unsqueeze(0)

# Function to generate a response using the trained model
def generate_response(sentence, model, words, classes):
    sentence_words = preprocess_sentence(sentence, words)
    if len(sentence_words) == 0:
        return "I'm sorry, but I don't understand. Can you please rephrase or provide more information?"

    features = sentence_to_features(sentence_words, words)
    with torch.no_grad():
        outputs = model(features)

    probabilities, predicted_class = torch.max(outputs, dim=1)
    confidence = probabilities.item()
    predicted_tag = classes[predicted_class.item()]

    if confidence > 0.5:
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                return random.choice(intent['responses'])

    return "I'm sorry, but I'm not sure how to respond to that."

# Interactive Chatbot
model_path = 'model.pth'
model = load_model(model_path, input_size, hidden_size, output_size)

print('Hello! I am a chatbot. How can I help you today? Type "quit" to exit.')
while True:
    user_input = input('> ')
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input, model, words, classes)
    print(response)
