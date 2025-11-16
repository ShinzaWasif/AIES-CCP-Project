# Hospital Recommendation Chatbot

A project that uses Natural Language Processing to help users find hospitals based on their medical needs, location, and budget.

## Team Members
- Syeda Shinza Wasif (CT-22063)
- Areeba Batool (CT-22059)
- Asifa Siraj (CT-22070)


## Overview
This chatbot understands natural language queries and recommends suitable hospitals. Users can type or speak their requirements, and the system provides relevant results with voice responses.

**Example Queries:**
- "heart specialist in karachi"
- "dentist under 5000"
- "eye doctor lahore 3000-5000"

## NLP Techniques Used

### 1. **Tokenization (NLTK)**
Breaks user queries into words for analysis
```python
query_tokens = word_tokenize(query.lower())
```

### 2. **Named Entity Recognition**
Extracts cities and specializations from queries
```python
def extract_city_from_query(query):
    # Matches city names from query tokens
```

### 3. **Pattern Matching (Regex)**
Identifies fee ranges and numbers
```python
range_match = re.search(r"(\d+)\s*-\s*(\d+)", query)
```

### 4. **Fuzzy String Matching**
Finds hospitals even with spelling mistakes or partial matches
```python
fuzzy_results = process.extract(query, specializations, limit=10)
```

### 5. **Speech Recognition (Wav2Vec 2.0)**
Converts audio input to text using Facebook's pre-trained model
```python
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
```

### 6. **Text-to-Speech**
Responds with voice output using gTTS and pyttsx3

### 7. **Number Conversion**
Handles both numeric and word-based inputs
- "five thousand" → 5000
- 5000 → "five thousand" (for TTS)

## Tech Stack

**Backend (Flask)**
- Flask API for handling requests
- NLTK for tokenization
- FuzzyWuzzy for fuzzy matching
- Transformers for speech recognition
- Beautiful Soup for web scraping hospital data

**Frontend (React)**
- User interface for chat
- Audio recording functionality
- Display hospital results

**Data Collection**
- Web scraping using Beautiful Soup
- Hospital data stored in JSON format

## How It Works

1. **User Input** → Query typed or spoken
2. **NLP Processing** → Extract city, specialization, fee range
3. **Fuzzy Matching** → Find similar hospitals (70% threshold)
4. **Filtering** → Apply location and budget filters
5. **Ranking** → Sort by fees (lowest first)
6. **Response** → Return results with voice output

## Key Features

- Natural language understanding
- Voice input/output support
- Fuzzy search for typos
- Multi-criteria filtering
- Real-time results

