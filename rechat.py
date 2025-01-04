import os
import csv
import datetime
import ssl
import streamlit as st
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL context for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from JSON file
with open('myintentb.json', 'r') as file:
    intents = json.load(file)

# Prepare data for training
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents['intents']:  #to access the 'intents' key in JSON
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents['intents']:  #  to access 'intents' key in JSON
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Main function for Streamlit
def main():
    st.title("Emotional Support Chatbot")
    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Hello, this is your emotional companion. Feel free to talk to me!")

        # Create chat log file if not exists
        if not os.path.exists("chat_log.csv"):
            with open("chat_log.csv", "w", newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # User input and chatbot response
        user_input = st.text_input("You", key="user_input")
        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key="chatbot_response")

            # Log conversation to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        with st.expander("Click to view conversation history"):
            if os.path.exists('chat_log.csv'):
                with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)  # Skip header
                    for row in csv_reader:
                        st.text(f"User: {row[0]}")
                        st.text(f"Chatbot: {row[1]}")
                        st.text(f"Timestamp: {row[2]}")
                        st.markdown("---")
            else:
                st.write("No conversation history found.")

    elif choice == "About":
        st.write("The goal of this project is to create a chatbot that understands and responds to user queries.")
        st.subheader("Project Overview:")
        st.write("""This chatbot uses Python, NLP techniques, and Logistic Regression for intent classification. 
                    It provides an interactive user interface with Streamlit.""")

        st.header("Key Features")
        st.write("""
        1. Utilizes NLP techniques to process user input.
        2. Employs Logistic Regression for intent classification.
        3. Provides a Streamlit-based web interface.
        4. Logs conversation history.
        5. Generates appropriate responses to user queries.
        """)

        st.header("Future Enhancements")
        st.write("""
        1. Add intent identification for better understanding.
        2. Incorporate emotion detection for empathetic responses.
        3. Support multiple languages.
        """)

if __name__ == '__main__':
    main()
