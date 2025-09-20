# backend_with_knn.py
from flask import Flask, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process
import json
from flask_cors import CORS
import re
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import numpy as np
from word2number import w2n
import time
import random
import os
import pygame
from gtts import gTTS
import threading
import inflect 
import pyttsx3
from collections import Counter

# NEW ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

nltk.download("punkt")
nltk.download("punkt_tab")

# Load hospital data
with open("hospitals.json", "r", encoding="utf-8") as file:
    hospital_data = json.load(file)

# Load Wav2Vec 2.0 model (if you need audio transcription)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def convert_numbers(transcription):
    words = transcription.lower()
    try:
        return str(w2n.word_to_num(words))
    except ValueError:
        return words

def extract_number_range(text):
    match = re.findall(r"(\d+[\s-]\d[\s-]\d)", text)
    return match if match else None

def transcribe_audio(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    max_length = 50
    chunk_size = sr * max_length
    transcriptions = []

    for i in range(0, len(speech), chunk_size):
        chunk = speech[i: i + chunk_size]
        input_values = processor(chunk, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions.append(processor.batch_decode(predicted_ids)[0])

    full_transcription = " ".join(transcriptions).lower()
    full_transcription = convert_numbers(full_transcription)
    number_ranges = extract_number_range(full_transcription)
    if number_ranges:
        full_transcription += " " + " ".join(number_ranges)

    return full_transcription

def extract_city_from_query(query):
    query_tokens = set(word_tokenize(query.lower()))
    possible_cities = {hospital["City"].lower() for hospital in hospital_data if "City" in hospital}
    for city in possible_cities:
        if city in query_tokens:
            return city
    return None

def extract_fee_from_query(query):
    range_match = re.search(r"(\d+)\s*-\s*(\d+)", query)
    single_match = re.search(r"(\d{4,6})", query)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))
    elif single_match:
        single_fee = int(single_match.group(1))
        return single_fee, single_fee
    return None, None

def is_within_fee_range(hospital_fee, min_fee, max_fee):
    match = re.search(r"(\d+)\s*-\s*(\d+)", hospital_fee)
    if match:
        hospital_min_fee, hospital_max_fee = int(match.group(1)), int(match.group(2))
        if min_fee == max_fee:
            return (hospital_min_fee <= max_fee <= hospital_max_fee) or (hospital_max_fee <= max_fee)
        return (hospital_min_fee >= min_fee and hospital_max_fee <= max_fee) or (hospital_min_fee == max_fee)
    return False

def find_matching_hospitals(query):
    query = query.lower().strip()
    query_tokens = set(word_tokenize(query))
    city_filter = extract_city_from_query(query)
    min_fee, max_fee = extract_fee_from_query(query)

    exact_matches = []
    fuzzy_candidates = []
    specializations_dict = {}

    for hospital in hospital_data:
        specialization = hospital.get("Specialization", "").lower()
        hospital_city = hospital.get("City", "").lower()
        hospital_fee = hospital.get("Fees", "")

        specializations_dict[specialization] = hospital
        specialization_tokens = set(word_tokenize(specialization))

        if query in specialization or query_tokens & specialization_tokens:
            if city_filter and hospital_city != city_filter:
                continue
            if min_fee is not None and not is_within_fee_range(hospital_fee, min_fee, max_fee):
                continue
            exact_matches.append(hospital)

    if not exact_matches:
        fuzzy_results = process.extract(query, specializations_dict.keys(), limit=10)
        for best_match, score in fuzzy_results:
            if score > 70:
                hospital = specializations_dict[best_match]
                hospital_city = hospital.get("City", "").lower()
                hospital_fee = hospital.get("Fees", "")
                if city_filter and hospital_city != city_filter:
                    continue
                if min_fee is not None and not is_within_fee_range(hospital_fee, min_fee, max_fee):
                    continue
                fuzzy_candidates.append(hospital)

    final_results = exact_matches + fuzzy_candidates
    try:
        return sorted(final_results, key=lambda h: int(re.search(r"(\d+)", h.get("Fees","0")).group(1))) if final_results else None
    except Exception:
        return final_results if final_results else None

p = inflect.engine()

def convert_numbers_to_words(text):
    def replace_number(match):
        num = int(match.group(0))
        return p.number_to_words(num, andword="")
    return re.sub(r"\b\d+\b", replace_number, text)

audio_lock = threading.Lock()

def speak(hospitals, user_fee_range=None, use_offline_tts=True):
    def tts_worker():
        try:
            if not hospitals:
                text = "Sorry, no matching hospitals found."
            else:
                hospital_info = []
                for hospital in hospitals:
                    name = hospital.get("Name", "Unknown Hospital")
                    city = hospital.get("City", "Unknown Location")
                    hospital_fees = hospital.get("Fees", "Not Available")
                    hospital_fees = re.sub(r"\b0+\b", "zero", hospital_fees)
                    hospital_fees = convert_numbers_to_words(hospital_fees)
                    user_fee_text = ""
                    if user_fee_range:
                        min_fee, max_fee = user_fee_range
                        user_fee_text = f"Your fee range is {convert_numbers_to_words(str(min_fee))} to {convert_numbers_to_words(str(max_fee))}."
                    hospital_text = f"{name}, located in {city}. Hospital fee range is {hospital_fees}. {user_fee_text}"
                    hospital_info.append(hospital_text)
                text = "Dear user, according to your search following are the hospitals. " + " ".join(hospital_info) + " For more, please visit the website."

            if use_offline_tts:
                engine = pyttsx3.init()
                voices = engine.getProperty("voices")
                indian_voice = None
                for voice in voices:
                    if "en-in" in voice.id.lower() or "india" in voice.name.lower():
                        indian_voice = voice.id
                        break
                if indian_voice:
                    engine.setProperty("voice", indian_voice)
                engine.setProperty("rate", 165)
                with audio_lock:
                    engine.say(text)
                    engine.runAndWait()
            else:
                tts = gTTS(text, lang="en", tld="co.in")
                temp_filename = f"temp_audio_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
                temp_path = os.path.join(os.getcwd(), temp_filename)
                tts.save(temp_path)
                with audio_lock:
                    pygame.mixer.init()
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.delay(500)
                    pygame.mixer.quit()
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except PermissionError:
                        print("Could not delete temp file immediately.")
        except Exception as e:
            print(f"Error in speak function: {e}")

    threading.Thread(target=tts_worker).start()

# ----------------- Recommender section (TF-IDF + kNN + clustering) -----------------

def make_text_by_basis(h, basis="auto"):
    if basis == "specialization":
        return (h.get("Specialization", "") or "").lower()
    if basis == "city":
        return (h.get("City", "") or "").lower()
    parts = []
    for key in ["Name", "City", "Province", "Specialization", "Address"]:
        val = h.get(key, "")
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    parts.append(str(h.get("Fees", "")))
    return " ".join(parts).lower()

_BASIS_LIST = ["auto", "specialization", "city"]
_tfidf_store = {}       # basis -> (vectorizer, tfidf_matrix)
_nn_store = {}          # basis -> NearestNeighbors fitted (k-NN)
_kmeans_store = {}      # basis -> KMeans fitted
_cluster_assignments = {}  # basis -> cluster ids

# clustering hyperparam
_DEFAULT_N_CLUSTERS = 8

for basis in _BASIS_LIST:
    corpus = [make_text_by_basis(h, basis=basis) for h in hospital_data]
    vec = TfidfVectorizer(stop_words="english")
    if len(corpus) == 0:
        mat = None
    else:
        mat = vec.fit_transform(corpus)
    _tfidf_store[basis] = (vec, mat)

    # fit NearestNeighbors (k-NN) using cosine metric
    if mat is not None and mat.shape[0] > 0:
        try:
            nn = NearestNeighbors(n_neighbors=min(20, mat.shape[0]), metric="cosine", algorithm="brute")
            nn.fit(mat)
            _nn_store[basis] = nn
        except Exception as e:
            print(f"Warning: could not fit NearestNeighbors for basis={basis}: {e}")
            _nn_store[basis] = None
        # fit KMeans clustering (optional)
        try:
            n_clusters = min(_DEFAULT_N_CLUSTERS, max(2, mat.shape[0] // 4))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(mat)
            _kmeans_store[basis] = kmeans
            _cluster_assignments[basis] = kmeans.labels_
        except Exception as e:
            print(f"Warning: could not fit KMeans for basis={basis}: {e}")
            _kmeans_store[basis] = None
            _cluster_assignments[basis] = None
    else:
        _nn_store[basis] = None
        _kmeans_store[basis] = None
        _cluster_assignments[basis] = None

def recommend_hospitals_by_basis(history, basis="auto", top_k=5, filter_city=None, fee_range=None, method="cosine"):
    """
    history: list of strings (user search history)
    basis: "auto" | "specialization" | "city"
    method: "cosine" | "knn" | "cluster"
    """
    if not history:
        return []
    basis = basis if basis in _BASIS_LIST else "auto"
    vectorizer, tfidf_matrix = _tfidf_store.get(basis, (None, None))
    if tfidf_matrix is None:
        return []
    user_doc = " ".join(history).lower()
    user_vec = vectorizer.transform([user_doc])

    # choose method
    indices = []
    if method == "knn":
        nn = _nn_store.get(basis)
        if nn is None:
            sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
            indices = sims.argsort()[::-1][:top_k].tolist()
        else:
            try:
                distances, neigh_idx = nn.kneighbors(user_vec, n_neighbors=min(top_k, tfidf_matrix.shape[0]))
                # kneighbors returns distances based on cosine distance; neigh_idx is neighbor indices
                indices = neigh_idx.flatten().tolist()
            except Exception:
                sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
                indices = sims.argsort()[::-1][:top_k].tolist()

    elif method == "cluster":
        kmeans = _kmeans_store.get(basis)
        if kmeans is None:
            sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
            indices = sims.argsort()[::-1][:top_k].tolist()
        else:
            try:
                user_cluster = kmeans.predict(user_vec)[0]
                cluster_ids = [i for i, lbl in enumerate(_cluster_assignments[basis]) if lbl == user_cluster]
                if not cluster_ids:
                    sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
                    indices = sims.argsort()[::-1][:top_k].tolist()
                else:
                    # compute sims but only for cluster members
                    cluster_mat = tfidf_matrix[cluster_ids]
                    sims = cosine_similarity(user_vec, cluster_mat).flatten()
                    ranked_pos = sims.argsort()[::-1][:top_k]
                    indices = [cluster_ids[i] for i in ranked_pos]
            except Exception:
                sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
                indices = sims.argsort()[::-1][:top_k].tolist()
    else:  # cosine default
        sims = cosine_similarity(user_vec, tfidf_matrix).flatten()
        indices = sims.argsort()[::-1][:top_k].tolist()

    # apply optional filters and build final list
    final = []
    for idx in indices:
        if len(final) >= top_k:
            break
        h = hospital_data[idx]
        if filter_city and h.get("City", "").strip().lower() != filter_city.strip().lower():
            continue
        if fee_range:
            try:
                min_fee, max_fee = fee_range
                hospital_fee_str = h.get("Fees", "")
                m = re.search(r"(\d+)\s*-\s*(\d+)", str(hospital_fee_str))
                if m:
                    hmin, hmax = int(m.group(1)), int(m.group(2))
                else:
                    m2 = re.search(r"(\d+)", str(hospital_fee_str))
                    if m2:
                        hmin = hmax = int(m2.group(1))
                    else:
                        continue
                if not (hmax >= min_fee and hmin <= max_fee):
                    continue
            except Exception:
                continue
        final.append(h)

    # if not enough after filtering, fill with top from cosine ranking
    if len(final) < top_k:
        sims_all = cosine_similarity(user_vec, tfidf_matrix).flatten()
        ranked_all = sims_all.argsort()[::-1]
        for idx in ranked_all:
            if len(final) >= top_k:
                break
            h = hospital_data[idx]
            if h in final:
                continue
            if filter_city and h.get("City", "").strip().lower() != filter_city.strip().lower():
                continue
            if fee_range:
                try:
                    min_fee, max_fee = fee_range
                    hospital_fee_str = h.get("Fees", "")
                    m = re.search(r"(\d+)\s*-\s*(\d+)", str(hospital_fee_str))
                    if m:
                        hmin, hmax = int(m.group(1)), int(m.group(2))
                    else:
                        m2 = re.search(r"(\d+)", str(hospital_fee_str))
                        if m2:
                            hmin = hmax = int(m2.group(1))
                        else:
                            continue
                    if not (hmax >= min_fee and hmin <= max_fee):
                        continue
                except Exception:
                    continue
            final.append(h)
    return final[:top_k]

@app.route("/recommend", methods=["POST"])
def recommend_route():
    data = request.get_json() or {}
    history = data.get("history", []) or []
    basis = data.get("basis", "auto")
    top_k = int(data.get("top_k", 5))
    method = data.get("method", "cosine")  # "cosine" | "knn" | "cluster"
    filter_city = data.get("filter_city", None)
    fee_range = None
    if "fee_range" in data and isinstance(data.get("fee_range"), (list, tuple)) and len(data.get("fee_range")) == 2:
        try:
            fee_range = (int(data["fee_range"][0]), int(data["fee_range"][1]))
        except Exception:
            fee_range = None

    results = recommend_hospitals_by_basis(history, basis=basis, top_k=top_k, filter_city=filter_city, fee_range=fee_range, method=method)
    return jsonify({"recommendations": results})

# Small metrics endpoint (optional)
@app.route("/recommend_metrics", methods=["GET"])
def recommend_metrics():
    info = {}
    for basis in _BASIS_LIST:
        vec, mat = _tfidf_store.get(basis, (None, None))
        info[basis] = {
            "n_hospitals": 0 if mat is None else int(mat.shape[0]),
            "n_features": 0 if mat is None else int(mat.shape[1]),
            "has_knn": basis in _nn_store and _nn_store[basis] is not None,
            "has_kmeans": basis in _kmeans_store and _kmeans_store[basis] is not None,
        }
    return jsonify(info)

print("=== Recommender: TF-IDF + k-NN + KMeans prepared for bases:", _BASIS_LIST)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_query = data.get("query", "").strip()
    audio_file = data.get("audio")
    if audio_file:
        transcription = transcribe_audio(audio_file)
        user_query = transcription
    if not user_query:
        return jsonify({"response": "Please provide a valid query."})
    matching_hospitals = find_matching_hospitals(user_query)
    if matching_hospitals:
        response_json = jsonify({"response": matching_hospitals})
        threading.Thread(target=speak, args=(matching_hospitals,)).start()
        return response_json
    else:
        return jsonify({"response": "Sorry, no matching hospitals found."})

if __name__ == "__main__":
    app.run(debug=True)
