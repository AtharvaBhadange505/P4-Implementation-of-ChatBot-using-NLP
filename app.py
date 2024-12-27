import os  
import json  
import datetime  
import csv  
import nltk  
import ssl  
import random  
import streamlit as st  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.linear_model import LogisticRegression  

# Download NLTK data  
ssl._create_default_https_context = ssl._create_unverified_context  
nltk.download('punkt')  

# Load intents from the JSON file  
file_path = os.path.abspath("./intents.json")  
with open(file_path, "r") as file:  
    intents = json.load(file)  

# Create the vectorizer and classifier  
vectorizer = TfidfVectorizer()  
clf = LogisticRegression(random_state=0, max_iter=10000)  

# Preprocess the data  
tags = []  
patterns = []  
for intent in intents['intents']:  
    for pattern in intent['patterns']:  
        tags.append(intent['tag'])  
        patterns.append(pattern)  

# Train the model  
x = vectorizer.fit_transform(patterns)  
y = tags  
clf.fit(x, y)  

def main():  
    st.title("Chatbot")  
    user_input = st.text_input("You: ")  

    if st.button("Send"):  
        # Predict the intent  
        user_input_vectorized = vectorizer.transform([user_input])  
        intent_index = clf.predict(user_input_vectorized)[0]  
        intent_response = next(intent['responses'] for intent in intents['intents'] if intent['tag'] == intent_index)  

        # Log the conversation  
        with open('chat_log.csv', 'a', newline="", encoding='utf-8') as f:  
            writer = csv.writer(f)  
            writer.writerow([user_input, random.choice(intent_response), datetime.datetime.now()])  

        st.text(f"Chatbot: {random.choice(intent_response)}")  

    # Conversation History Menu  
    if st.button("Conversation History"):  
        st.header("Conversation History")  
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:  
            csv_reader = csv.reader(csvfile)  
            next(csv_reader)  # Skip the header row  
            for row in csv_reader:  
                st.text(f"User: {row[0]}")  
                st.text(f"Chatbot: {row[1]}")  
                st.text(f"Timestamp: {row[2]}")  
                st.markdown("---")  

if __name__ == "__main__":  
    main()