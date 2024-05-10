from flask import Flask, request, jsonify
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
questionsEN = []
answersEN = []
with open('/home/n4h0/mysite/Q&AEnglish.txt', 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith('Q:'):
            questionsEN.append(line[3:].strip())
        if line.startswith('A') and not line.startswith('AF'):
            answersEN.append(line[3:].strip())

#Henter den encoda lista med spørsmål i json format
with open('/home/n4h0/mysite/Q&A_embeddedEnglish.json', 'r', encoding='utf-8') as file:
    loaded_list_as_listsEN2 = json.load(file)

#Konverterer den encoda lista til ei liste med arrays.
def convert_to_arrays(loaded_list_as_listsEN):
    return [np.array(sublist) for sublist in loaded_list_as_listsEN]

encoded_questions_listEN = convert_to_arrays(loaded_list_as_listsEN2)


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
with open('/home/n4h0/mysite/Q&A_embeddedetFitModel.json', 'r', encoding='utf-8') as file:
    loaded_list_as_lists2 = json.load(file)

#Konverterer den encoda lista til ei liste med arrays.
def convert_to_arrays(loaded_list_as_lists):
    return [np.array(sublist) for sublist in loaded_list_as_lists]

encoded_questions_list = convert_to_arrays(loaded_list_as_lists2)

#Henter modellen
model_name = "DiffuseCalmly/BachelorSBERT"
model = SetFitModel.from_pretrained(model_name)

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'Missing question in request'}), 400

    user_question = request.json['question']
    user_language = request.json.get('language', 'norsk')

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user_questions (question) VALUES (?)', (user_question,))
    conn.commit()
    conn.close()

    encoded_user_question = model.encode([user_question])[0]
    similarity_scores = []

    if user_language == 'english':
        questions_list = questionsEN
        answers_list = answersEN
        encoded_questions_listen = encoded_questions_listEN
        mel1 = "Did you mean to ask any of these questions?"
        mel2 = 'Sorry, I did not understand the question. Try asking in another way, or ask one of these questions:'

    else:
        questions_list = questions
        answers_list = answers
        encoded_questions_listen = encoded_questions_list
        mel1 = "Jeg forsto ikke spørsmålet, prøv å omformulere det eller velg ett av disse alternativene."
        mel2 = 'Beklager, jeg forsto ikke spørsmålet. Prøv å stille spørsmålet på en annen måte, eller still meg ett av disse spørsmålene'


    for sublist in encoded_questions_listen:
        similarity_scores.append(max(cosine_similarity([encoded_user_question], sublist)[0]))

    top_indices = np.argsort(similarity_scores)[-3:][::-1]  # Top 3 indices
    CoSimScore = max(similarity_scores)

    if CoSimScore > 0.80:
        response = answers_list[top_indices[0]]
        return jsonify(response)
    elif CoSimScore > 0.70:
        question_suggestions = [{'question': questions_list[i], 'score': similarity_scores[i]} for i in top_indices]
        return jsonify({
            'message': answers_list[top_indices[0]] + '\n' + mel1,
            'suggestions': question_suggestions
        })
    else:
        commonQuestions = [{'question': questions_list[i]} for i in [28,31,35]]
        return jsonify({
            'message':mel2,
            'suggestions': commonQuestions
        })

if __name__ == '__main__':
    app.run(debug=True)
