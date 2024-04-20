from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS
import json
import torch
import sqlite3

app = Flask(__name__)
CORS(app)

#Denne må være med
torch.set_num_threads(1)

def get_db_connection():
    conn = sqlite3.connect('/home/n4h0/mysite/chatbot.db')
    conn.row_factory = sqlite3.Row  # Gjør at du kan referere til kolonner ved navn
    return conn

# Henter alle spørsmål og svar
questions = []
answers = []
with open('/home/n4h0/mysite/Q&A.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questions.append(line[3:].strip())
        if line.startswith('A') and not line.startswith('AF'):
            answers.append(line[3:].strip())

print(questions)

#Henter den encoda lista med spørsmål i json format
with open('/home/n4h0/mysite/Q&A_embedded.json', 'r', encoding='utf-8') as file:
    loaded_list_as_lists = json.load(file)

#Konverterer den encoda lista til ei liste med arrays.
def convert_to_arrays(loaded_list_as_lists):
    return [np.array(sublist) for sublist in loaded_list_as_lists]

encoded_questions_list = convert_to_arrays(loaded_list_as_lists)

#Henter modellen
model_name = "NbAiLab/nb-sbert-base"
model = SentenceTransformer(model_name)

#Sjølve apien
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'Missing question in request'}), 400

    #Henter spørsmålet til bruker
    user_question = request.json['question']

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user_questions (question) VALUES (?)', (user_question,))
    conn.commit()
    conn.close()

    #Encoder spørsmålet til bruker
    encoded_user_question = model.encode([user_question])[0]
    similarity_scores = []
    #Gjer det på denne måten slik at einaste verdien som blir lagra er maksverdien av Q og alle AF til Q.
    for sublist in encoded_questions_list:
        similarity_scores.append(max(cosine_similarity([encoded_user_question], sublist)[0]))
    most_similar_question_index = np.argmax(similarity_scores)

    nested_list = []

    for question, similarity_score in zip(questions, similarity_scores):
        sublist = [question, float(similarity_score)]
        nested_list.append(sublist)

    CoSimScore = max(similarity_scores)

    if CoSimScore > 0.50:
        most_similar_question = answers[most_similar_question_index]
        print("Returning output to user: ", most_similar_question)
        return jsonify(f"{most_similar_question} CoSim = {CoSimScore}")
    else:
        return jsonify(f"Error, can't answer that. CoSim = {CoSimScore}")

if __name__ == '__main__':
    app.run(debug=True)