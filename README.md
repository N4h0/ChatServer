# Q&A chatbot

## Beskrivelse

Dette repositoriet blir brukt til å kjøre en server i PythonAnywhere. Serveren sammenligner et input-spørsmål med spørsmålene i Q&A.txt eller Q&AEnglish.txt or returnerer det likeste spørsmålet. Chatbotten bruker en fine-tuna versjon av modellen  https://huggingface.co/NbAiLab/nb-sbert-base (https://huggingface.co/DiffuseCalmly/BachelorSBERT). 

##Bruk

Chatbotten er laget for å kjøre på PythonAnywhere, ved å laste opp alle filer og starte en server. 

## Innhold
encode_questions.py: Skript som brukes for å lage embeddings for spørsmålene i Q&A-filene.
flask_app.py: Flask-applikasjon som håndterer API-forespørsler og kjører likhetssammenligninger mellom brukerens spørsmål og de forhåndsdefinerte spørsmålene.
Q&A.txt: Fil som inneholder spørsmål og svar på norsk.
Q&AEnglish.txt: Fil som inneholder spørsmål og svar på engelsk.

Flask-applikasjon
flask_app.py er hovedskriptet som kjører Flask-serveren. Den benytter seg av SetFitModel for å kode brukerens spørsmål og sammenligne det med de forhåndskodede spørsmålene ved hjelp av cosine-similarity.


For å kjøre denne applikasjonen trenger du følgende pakker:

Flask
scikit-learn
numpy
pytorch
setfit
flask_cors
sqlite3
