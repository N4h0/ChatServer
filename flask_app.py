from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_cors import CORS
import json
import torch
import sqlite3
from setfit import SetFitModel

app = Flask(__name__)
CORS(app)

#Denne linja er viktig! https://help.pythonanywhere.com/pages/MachineLearningInWebsiteCode
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

#Henter den encoda lista med spørsmål i json format
with open('/home/n4h0/mysite/Q&A_embedded.json', 'r', encoding='utf-8') as file:
    loaded_list_as_lists = json.load(file)

with open('/home/n4h0/mysite/Q&A_embeddedetFitModel.json', 'r', encoding='utf-8') as file:
    loaded_list_as_lists2 = json.load(file)

#Konverterer den encoda lista til ei liste med arrays.
def convert_to_arrays(loaded_list_as_lists):
    return [np.array(sublist) for sublist in loaded_list_as_lists]

encoded_questions_list = convert_to_arrays(loaded_list_as_lists)
encoded_questions_list2 = convert_to_arrays(loaded_list_as_lists2)

#Henter modellen
model_name = "NbAiLab/nb-sbert-base"
model = SentenceTransformer(model_name)

second_model_name = "DiffuseCalmly/BachelorSBERT"
model2 = SetFitModel.from_pretrained(second_model_name)

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
    encoded_user_question2 = model2.encode([user_question])[0]

    similarity_scores = []
    similarity_scores2 = []
    #Gjer det på denne måten slik at einaste verdien som blir lagra er maksverdien av Q og alle AF til Q.
    for sublist in encoded_questions_list:
        similarity_scores.append(max(cosine_similarity([encoded_user_question], sublist)[0]))
    for sublist in encoded_questions_list2:
        similarity_scores2.append(max(cosine_similarity([encoded_user_question2], sublist)[0]))
    top_indices = np.argsort(similarity_scores)[-3:][::-1] #Returnerer omdexem tol dei tre høgste verdiane
    top_indices2 = np.argsort(similarity_scores)[-3:][::-1] #Returnerer omdexem tol dei tre høgste verdiane

    CoSimScore = max(similarity_scores)
    CoSimScore2 = max(similarity_scores2)

    if CoSimScore > 0.50:
        print("Returning output to user: ", answers[top_indices[0]])
        return jsonify(f"{answers[top_indices[0]]} CoSim = {CoSimScore} \n {answers[top_indices2[0]]} CoSim2 = {CoSimScore2}")
    elif CoSimScore > 0.30:
        question_suggestions_with_scores = [
            {'question': questions[i], 'CoSim': similarity_scores2[i]}
            for i in top_indices2
        ]
        return jsonify({
            'message': f'Mente du et av disse spørsmålene? \n {answers[top_indices2[0]]} CoSim2 = {CoSimScore2}',
            'suggestions': question_suggestions_with_scores
        })
    else:
        return jsonify({'error': "Beklager, ingen tilstrekkelige svar funnet. Still gjerne spørsmålet på en annen måte."})


if __name__ == '__main__':
    app.run(debug=True)