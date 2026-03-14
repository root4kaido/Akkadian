"""exp015: last_model でのCV評価"""
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
# last_modelのパスを環境変数で渡してからeval_cv相当を実行
os.environ["EXP015_MODEL_OVERRIDE"] = os.path.join(EXP_DIR, "results", "last_model")
# eval_cv.pyを直接importせず、必要な部分だけ実行
import re, json, math, logging
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml, sacrebleu
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "last_model")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "workspace", "exp007_mbr_postprocess", "src"))
from infer_mbr import repeat_cleanup

config_path = os.path.join(EXP_DIR, "config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)
MAX_LENGTH = config["model"]["max_length"]
SEED = config["training"]["seed"]
PREFIX = config["model"]["prefix"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = os.path.join(EXP_DIR, config["data"]["train_path"])
df = pd.read_csv(train_path)
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]
_, val_split = train_test_split(df, test_size=config["training"]["val_ratio"], random_state=SEED)

dict_path = os.path.join(EXP_DIR, "dataset", "form_type_dict.json")
with open(dict_path) as f:
    form_tag_dict = json.load(f)

def tag_transliteration(text, ftd):
    return " ".join(f"{t}[{ftd[t]}]" if t in ftd else t for t in text.split())

def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

def truncate_akkadian_to_sentence(translit, max_bytes=200):
    enc = str(translit).encode('utf-8')
    if len(enc) <= max_bytes: return str(translit)
    trunc = enc[:max_bytes].decode('utf-8', errors='ignore')
    last = trunc.rfind(' ')
    return trunc[:last].strip() if last > 0 else trunc.strip()

sent_inputs, sent_refs = [], []
for _, row in val_split.iterrows():
    eng = extract_first_sentence(str(row["translation"]))
    akk = tag_transliteration(truncate_akkadian_to_sentence(str(row["transliteration"])), form_tag_dict)
    if eng.strip() and akk.strip():
        sent_inputs.append(PREFIX + akk)
        sent_refs.append(eng)

logger.info(f"Sent-level val: {len(sent_inputs)}, Model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

class InferenceDataset(TorchDataset):
    def __init__(self, texts, tok, ml):
        self.enc = tok(texts, max_length=ml, truncation=True, padding="max_length", return_tensors="pt")
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}

loader = DataLoader(InferenceDataset(sent_inputs, tokenizer, MAX_LENGTH), batch_size=4, shuffle=False)
preds = []
with torch.no_grad():
    for batch in tqdm(loader, desc="CV eval last_model (beam4)"):
        out = model.generate(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                             max_length=MAX_LENGTH, num_beams=4, early_stopping=True)
        preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])

preds_clean = [repeat_cleanup(p) for p in preds]

def has_repetition(text, mr=3):
    w = str(text).split()
    for i in range(len(w)-mr):
        if " ".join(w[i:i+mr]) in " ".join(w[i+mr:]): return True
    return False

for plabel, pl in [("raw", preds), ("clean", preds_clean)]:
    chrf = sacrebleu.corpus_chrf(pl, [sent_refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(pl, [sent_refs])
    geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
    rep = 100 * sum(has_repetition(p) for p in pl) / len(pl)
    logger.info(f"  {plabel:5s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep:.1f}%")
