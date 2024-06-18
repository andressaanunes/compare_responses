from modulefinder import Module
import openai
import os
import pandas as pd
import numpy as np
import traceback
import warnings
import torch
from flask import Flask, request, render_template, redirect, url_for
from getpass import getpass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from config import Config


app = Flask(__name__)
openai.api_key = getpass(Config.OpenAI.API_KEY)

# warnings.filterwarnings("ignore", category=FutureWarning)

def compare_responses(file1, file2):
    # Verifique a extensão dos arquivos
    if not (file1.filename.lower().endswith(('.csv', '.xlsx')) and file2.filename.lower().endswith(('.csv', '.xlsx'))):
        raise ValueError("Unsupported file format")

    # Read the first file
    if file1.filename.lower().endswith('.csv'):
        df1 = pd.read_csv(file1)
    else:
        df1 = pd.read_excel(file1)

    # Read the second file
    if file2.filename.lower().endswith('.csv'):
        df2 = pd.read_csv(file2)
    else:
        df2 = pd.read_excel(file2)


    # Get response columns
    respostas_df1 = df1['Respostas'].fillna('').astype(str)
    respostas_df2 = df2['Respostas'].fillna('').astype(str)
    
    # print(respostas_df1)
    # print(respostas_df2)

    # Load pre-trained embedding model
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    # print(model)

    # Generate embeddings for responses
    embeddings_df1 = model.encode(respostas_df1, convert_to_tensor=True)
    embeddings_df2 = model.encode(respostas_df2, convert_to_tensor=True)
    print(embeddings_df1)
    print(embeddings_df2)
    
    # Salve os embeddings para uso futuro
    torch.save(embeddings_df1, 'embeddings_df1.pt')
    torch.save(embeddings_df2, 'embeddings_df2.pt')
    
    # Carregue os embeddings pré-calculados
    embeddings_df1 = torch.load('embeddings_df1.pt')
    embeddings_df2 = torch.load('embeddings_df2.pt')
    
    # Calcule a similaridade de cosseno em lote
    similarity_scores = cosine_similarity(embeddings_df1, embeddings_df2)

    # Converta a matriz de similaridade de cosseno em uma lista de tuplas (i, j, score)
    similarity_scores = [(i, j, similarity_scores[i, j]) for i in range(len(respostas_df1)) for j in range(len(respostas_df2)) if i != j]

    # Classifique os scores de similaridade por score em ordem decrescente
    sorted_scores = sorted(similarity_scores, key=lambda x: x[2], reverse=True)

    # Print top 10 most similar response pairs
    output = []
    for i, (df1_index, df2_index, score) in enumerate(sorted_scores[:10]):
        pair_output = {
            "pair": i+1,
            "file1": os.path.splitext(file1.filename)[0],
            "response1": respostas_df1.iloc[df1_index],
            "file2": os.path.splitext(file2.filename)[0],
            "response2": respostas_df2.iloc[df2_index],
            "score": float(score)
        }
        output.append(pair_output)
    return {"status": "success", "result": output}, 200

@app.route('/upload', methods=['POST'])
def get_input():
    file1 = request.files['file1']
    file2 = request.files['file2']
    print('Teste', file1, file2)
    
    try:
        compare_responses(file1, file2)
        return redirect(url_for('/output'))
        # return redirect(url_for('/output'))
        # return {"status": "success"}, 200
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}, 400
    
@app.route('/output', methods=['GET'])
def home():
    return render_template('promptfooconfig.yaml')

if __name__ == "__main__":
        app.run(port=8000)
