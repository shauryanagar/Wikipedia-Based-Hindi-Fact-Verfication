import os
import re
import requests
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForSequenceClassification, 
    pipeline as hf_pipeline
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = "model_weights.pt"

session = requests.Session()
session.headers.update({
    "User-Agent": "IndicFactVerificationSystem/1.0 (academic.indicnlp@gmail.com) python-requests"
})

DEBUNK_MARKERS = [
    "काल्पनिक", "भ्रम", "अफवाह", "गलत दावा", "मिथक", "झूठ",
    "fanciful belief", "myth", "hoax", "popular misconception", "discredited"
]

ner_name = "cfilt/HiNER-collapsed-muril-base-cased"
ner_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
ner_model = AutoModelForTokenClassification.from_pretrained(ner_name).to(DEVICE)
ner_pipe = hf_pipeline(
    "ner", 
    model=ner_model, 
    tokenizer=ner_tokenizer, 
    aggregation_strategy="simple", 
    device=0 if torch.cuda.is_available() else -1
)

sts_name = "l3cube-pune/hindi-sentence-similarity-sbert"
sts_embedder = SentenceTransformer(sts_name, device=DEVICE)

nli_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(DEVICE)

if os.path.exists(WEIGHTS_PATH):
    try:
        nli_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    except Exception:
        pass

nli_model.eval()

def extract_entities(claim: str) -> list:
    clean = re.sub(r'[।॥\.\?\!\"\'\,]', '', claim).strip()
    candidates = []
    
    try:
        results = ner_pipe(claim)
        for r in results:
            w = r.get('word', '').replace('##', '').strip()
            if len(w) >= 3 and w not in candidates:
                candidates.append(w)
    except Exception:
        pass
    
    words = clean.split()
    stops = {'है', 'हैं', 'था', 'थी', 'थे', 'का', 'की', 'के', 'में', 'पर', 'से', 'को', 'ने', 'और', 'या'}
    content_words = [w for w in words if w not in stops and len(w) >= 3]
    if len(content_words) >= 2:
        candidates.append(" ".join(content_words[:2]))
    for cw in content_words:
        if cw not in candidates:
            candidates.append(cw)
            
    candidates.append(clean)
    return list(dict.fromkeys(candidates))

def search_wikipedia(query: str, limit: int = 5) -> list:
    url = "https://hi.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "utf8": 1, "srlimit": limit}
    try:
        r = session.get(url, params=params, timeout=6)
        hits = [x["title"] for x in r.json().get("query", {}).get("search", [])]
        banned = ['संधि (व्याकरण)', 'संयुक्त व्यंजन', 'माहेश्वर सूत्र', 'व्याकरण']
        return [h for h in hits if not any(b in h for b in banned)]
    except Exception:
        return []

def get_wikipedia_corpus(titles: list) -> str:
    if not titles:
        return ""
    url = "https://hi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": "|".join(titles[:5]),
        "format": "json",
        "utf8": 1,
        "redirects": 1
    }
    try:
        r = session.get(url, params=params, timeout=10)
        pages = r.json().get("query", {}).get("pages", {})
        corpus = []
        for _, p in pages.items():
            extract = p.get("extract", "")
            if extract:
                corpus.append(extract)
        return " ".join(corpus)
    except Exception:
        return ""

def clean_and_split(text: str) -> list:
    if not text:
        return []
    text = re.sub(r'==+[^=]+==+', ' ', text)
    text = re.sub(r'Archived[^\n]+Wayback Machine', ' ', text, flags=re.IGNORECASE)
    raw_sentences = re.split(r'[।॥\n\r]+|(?<=[.?!])\s+', text)
    clean_sents = []
    for s in raw_sentences:
        s_clean = re.sub(r'\s+', ' ', s).strip()
        if len(s_clean) >= 15:
            clean_sents.append(s_clean)
    return clean_sents

def rank_sentences_mse(claim: str, sentences: list, top_k: int = 100) -> list:
    if not sentences:
        return []
    claim_vec = sts_embedder.encode(claim, convert_to_tensor=True)
    sent_vecs = sts_embedder.encode(sentences, convert_to_tensor=True)
    mse_losses = torch.mean((sent_vecs - claim_vec) ** 2, dim=-1)
    sorted_idx = torch.argsort(mse_losses, descending=False)
    return [sentences[i] for i in sorted_idx[:min(top_k, len(sentences))].cpu().tolist()]

def verify_claim(claim: str):
    entities = extract_entities(claim)
    
    all_titles = []
    for ent in entities[:3]:
        all_titles.extend(search_wikipedia(ent, limit=5))

    all_titles = list(dict.fromkeys(all_titles))
    if not all_titles:
        return {
            "claim": claim, 
            "verdict": "NOT ENOUGH INFO", 
            "confidence": "0.0%", 
            "sources": [], 
            "evidence": ["विकिपीडिया पर कोई संबंधित लेख नहीं मिला।"]
        }

    corpus = get_wikipedia_corpus(all_titles)
    sentences = clean_and_split(corpus)
    
    if not sentences:
        return {
            "claim": claim, 
            "verdict": "NOT ENOUGH INFO", 
            "confidence": "0.0%", 
            "sources": all_titles, 
            "evidence": ["स्रोतों से पठनीय साक्ष्य वाक्य प्राप्त नहीं हो सके।"]
        }

    top_100 = rank_sentences_mse(claim, sentences, top_k=100)

    best_verdict = "NOT ENOUGH INFO"
    best_confidence = 0.0
    best_sentence = top_100[0]

    for sent in top_100[:15]:
        if any(m in sent.lower() for m in DEBUNK_MARKERS):
            return {
                "claim": claim,
                "entities": entities,
                "verdict": "REFUTES (False)",
                "confidence": "96.5%",
                "sources": all_titles[:3],
                "evidence": [sent]
            }

        inputs = nli_tokenizer(sent, claim, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        entail = probs[0]
        contra = probs[2]

        if contra > entail and contra > 0.45:
            if contra > best_confidence:
                best_confidence = contra
                best_verdict = "REFUTES (False)"
                best_sentence = sent
        elif entail > contra and entail > 0.45:
            if entail > best_confidence:
                best_confidence = entail
                best_verdict = "SUPPORTS (True)"
                best_sentence = sent

    if best_confidence == 0.0:
        best_confidence = 0.50

    return {
        "claim": claim,
        "entities": entities,
        "verdict": best_verdict,
        "confidence": f"{round(float(best_confidence) * 100, 1)}%",
        "sources": all_titles[:3],
        "evidence": [best_sentence]
    }