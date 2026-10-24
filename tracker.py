"""Smart Calorie Tracker - Module 1 (food recognition with BLIP) and
Module 2 (nutrition estimation from a knowledge base with serving scaling).
Only ONE deep-learning model is used: BLIP."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = pd.read_csv(Path(__file__).resolve().parent / "kb.csv")
NUTR = ["kcal", "protein_g", "carbs_g", "fat_g"]

# ---------------- Module 1: image preprocessing + BLIP captioning ------------
_blip = None
def load_blip(name="Salesforce/blip-image-captioning-large"):
    global _blip
    if _blip is None:
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _blip = (BlipProcessor.from_pretrained(name),
                 BlipForConditionalGeneration.from_pretrained(name).to(dev).eval(), dev)
    return _blip

def preprocess(path, min_side=64):
    img = Image.open(path).convert("RGB")
    if min(img.size) < min_side:
        raise ValueError("image too small")
    return img

PROMPTS = ["", "a bowl of", "a plate of", "a meal with", "a healthy meal made of",
           "vegetables and", "rice and", "a photo of food containing"]

def caption(path, prompts=PROMPTS):
    """Run BLIP several times with different starting phrases (BLIP is the only model)
    and join the captions, because one sentence never names every food on the plate."""
    import torch
    proc, model, dev = load_blip()
    img = preprocess(path)
    caps = []
    for p in prompts:
        inp = proc(img, p, return_tensors="pt").to(dev) if p else proc(img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=30, num_beams=4)
        caps.append(proc.decode(out[0], skip_special_tokens=True))
    return " ; ".join(dict.fromkeys(caps))

# ---------------- Item extraction (maps text -> knowledge-base foods) --------
_alias = []
for _, r in KB.iterrows():
    for a in str(r["aliases"]).split(";") + [r["name"]]:
        _alias.append((a.strip().lower(), r["name"]))
_alias.sort(key=lambda x: -len(x[0]))          # longest alias first

def extract_items(text):
    """Return the list of knowledge-base foods mentioned in the text."""
    t = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    found = []
    for a, name in _alias:
        if f" {a} " in t or f" {a}s " in t:
            if name not in found:
                found.append(name)
            t = t.replace(a, " ")
    if not found:                               # fallback: TF-IDF nearest entry
        vec = TfidfVectorizer().fit(KB["aliases"].tolist() + [text])
        sims = cosine_similarity(vec.transform([text]), vec.transform(KB["aliases"]))[0]
        if sims.max() > 0.2:
            found.append(KB.iloc[int(sims.argmax())]["name"])
    return found

# ---------------- Module 2: nutrition estimation + portion scaling ----------
def estimate(items, servings=None):
    """items: list of food names; servings: dict name->number of servings (default 1).
    kb.csv values are PER SERVING (the serving size is in the 'serving' column)."""
    servings = servings or {}
    rows = []
    for n in items:
        r = KB[KB["name"] == n].iloc[0]
        k = float(servings.get(n, 1))
        rows.append({"item": n, "serving": r["serving"], "servings": k,
                     **{c: round(float(r[c]) * k, 1) for c in NUTR}})
    df = pd.DataFrame(rows)
    total = {c: round(float(df[c].sum()), 1) for c in NUTR} if len(df) else {c: 0 for c in NUTR}
    return df, total

def analyse(path, servings=None, text_hint=""):
    cap = caption(path)
    items = extract_items(cap + " " + text_hint)
    df, total = estimate(items, servings)
    return cap, items, df, total
