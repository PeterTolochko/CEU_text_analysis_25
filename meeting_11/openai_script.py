from pathlib import Path
from openai import OpenAI
import os

# openai api website
# https://platform.openai.com/docs/introduction



# Load API key from .env file
my_key = os.environ["GPT_KEY"]
client = OpenAI(api_key=my_key)



### GPT-3.5 Turbo CHAT BOT ###

# Chat with a model via the API
# This example shows a simple chat interface to a model.
my_chat_bot = "You are supposed to classify the sentiment of the text. You should classify the text as either positive or negative. Your answer should be only one of the following: positive OR negative."
query = "Oh wow. I really love ice-cream!" 


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role": "system", "content": my_chat_bot},
                {"role": "user", "content": query}],
  max_tokens=300,
)


print(response.choices[0].message.content)


# syntheica data generation with gpt turbo


my_chat_bot = "This is a fictional scenario. you are a human being living in the united states. your age is: 46 years old. your education level is university. your gender is male. You have to provide a political party preference. Your answer should be only one of the following: democrat OR republican."

query = "which political party would you vote for in the next election?" 


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages = [{"role": "system", "content": my_chat_bot},
                {"role": "user", "content": query}],
  max_tokens=50,
)


print(response.choices[0].message.content)


# synthetica data generation with gpt-4
# https://www.cambridge.org/core/journals/political-analysis/article/abs/out-of-one-many-using-language-models-to-simulate-human-samples/035D7C8A55B237942FB6DBAD7CAA4E49



# create different scenarios for synthetic data generation with gpt-4
# based on age, gender, education level

age = [18, 25, 35, 45, 55, 65, 75]
gender = ["male", "female"]
education = ["some high school", "high school", "university", "graduate school"]

# create a list with all possible combinations of age, gender, and education level

combinations = [(a, g, e) for a in age for g in gender for e in education]

political_preference = []
for combination in combinations:
    my_chat_bot = f"This is a fictional scenario. you are a human living in the united states. your age is: {combination[0]} years old. your education level is {combination[2]}. your gender is {combination[1]}. You have to provide a preference for the news source that you are using. Your answer should be only one of the following: CNN or Fox News."

    query = "which political party do you prefer? You have to chose between democrat or republican."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "system", "content": my_chat_bot},
                    {"role": "user", "content": query}],
    max_tokens=50,
    )
    political_preference.append(response.choices[0].message.content)
    print(response.choices[0].message.content)

# a function to extract the political preference from the response
# using regex
import re

def extract_party(text):
    return re.findall(r"[Dd]emocrat|[Rr]epublican", text, re.IGNORECASE)[0]

    
# extract the political preference from the response
political_preference_clean = []
for preference in political_preference:
  try:
    political_preference_clean.append(extract_party(preference))
  except:
    political_preference_clean.append("None")

political_preference_clean



# Text embedding with ada

# Text embeddings

response = client.embeddings.create(
    input="I love ice cream!",
    model="text-embedding-3-small"
)

len(response.data[0].embedding)
print(response.data[0].embedding)


def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding




# read the review data
import pandas as pd
path = "reviews_with_embeddings.csv"
df = pd.read_csv(path)

# sample for this example
df = df.sample(10)


## zero-shot classification with ada

# create a list of labels
labels = ["family", "horror"]
label_embeddings = [get_embedding(label, model="text-embedding-3-small") for label in labels]


# create a list of reviews
reviews = df["value"].tolist()

# create a list of embeddings (THIS IS VERY SLOW)
embeddings = [get_embedding(review) for review in reviews]

# add embeddings to the dataframe
# df["embedding"] = embeddings

# write to csv
# df.to_csv("reviews_with_embeddings.csv", index=False)

# read the embeddings
df = pd.read_csv("reviews_with_embeddings.csv")

embeddings = df["embedding"]
embeddings = [eval(embedding) for embedding in embeddings]

# function to calculate the cosine similarity between two vectors
import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def label_score(embedding, label_embeddings):
   return cosine_similarity(embedding, label_embeddings[1]) - cosine_similarity(embedding, label_embeddings[0])

# one prediction
prediction = 'family' if label_score(embeddings[0], label_embeddings) > 0 else 'horror'

# all predictions
predictions = ["family" if label_score(embeddings[i], label_embeddings) > 0 else "horror" for i in range(len(embeddings))]


# run logistic regression on the embeddings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = np.array(embeddings)
y = df["polarity"].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# run logistic regression
my_model = LogisticRegression(random_state=0).fit(X_train, y_train)
my_model

# predict on the test set
y_pred = my_model.predict(X_test)

# calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# calculate f1 score
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='macro')


# visualizing the embeddings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Apply t-SNE to reduce dimensionality to 2D
embeddings_array = np.array(embeddings)


tsne = TSNE(n_components=2,
            perplexity=30.0,
            random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

colors = ["red", "blue"]
color_labels = ["red" if df["polarity"].tolist()[i] == "negative" else "blue" for i in range(len(embeddings))]

colormap = matplotlib.colors.ListedColormap(colors)


plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0],
                      embeddings_2d[:, 1],
                      c=color_labels,
                      alpha=0.5,
                      cmap=colormap)
plt.title("t-SNE Visualization of Embeddings")
plt.show()




## Audio to text

# read the audio file

audio_file = open("sound_files/NPR3054741922.mp3", "rb")


transcript = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)

transcript




# image recognition

response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://image.smythstoys.com/zoom/219048.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])






images_path = Path("pictures")

# list files in the path
files = os.listdir(images_path)

import base64
import requests


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
# Getting the base64 string
# base64_image = encode_image(image_path)

my_images = [encode_image(images_path / file) for file in files]



headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {my_key}"
}

def generate_payload(current_image):
   return {
      "model": "gpt-4o-mini",
      "messages": [
         {"role": "user",
          "content": [
                {
                   "type": "text", "text": "Extract the text from the image."
                 },
                 {
                     "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{current_image}"
                        }
                 }
          ]}
      ],
        "max_tokens": 300
   }

for image in my_images:
    payload = generate_payload(image)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json()['choices'][0])

