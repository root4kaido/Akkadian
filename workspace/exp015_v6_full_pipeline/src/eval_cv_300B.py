"""exp015: 300B truncation でのCV評価（200Bとの比較用）"""
import os, sys, re, json, math, logging
import pandas as pd, numpy as np, torch, yaml, sacrebleu
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.abspath(os.path.join(EXP_DIR, "../.."))
RESULTS_DIR = os.path.join(EXP_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "workspace", "exp007_mbr_postprocess", "src"))
from infer_mbr import repeat_cleanup

with open(os.path.join(EXP_DIR, "config.yaml")) as f:
    config = yaml.safe_load(f)
MAX_LENGTH = config["model"]["max_length"]
SEED = config["training"]["seed"]
PREFIX = config["model"]["prefix"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(os.path.join(EXP_DIR, config["data"]["train_path"]))
df = df[(df["transliteration"].astype(str).str.len() > 0) & (df["translation"].astype(str).str.len() > 0)]
_, val_split = train_test_split(df, test_size=config["training"]["val_ratio"], random_state=SEED)

with open(os.path.join(EXP_DIR, "dataset", "form_type_dict.json")) as f:
    form_tag_dict = json.load(f)

def tag(text):
    return " ".join(f"{t}[{form_tag_dict[t]}]" if t in form_tag_dict else t for t in text.split())

def extract_first_sentence(text):
    m = re.search(r'^(.*?[.!?])(?:\s|$)', str(text))
    return m.group(1).strip() if m else str(text).strip()

def truncate(translit, max_bytes):
    enc = str(translit).encode('utf-8')
    if len(enc) <= max_bytes: return str(translit)
    trunc = enc[:max_bytes].decode('utf-8', errors='ignore')
    last = trunc.rfind(' ')
    return trunc[:last].strip() if last > 0 else trunc.strip()

def has_repetition(text, mr=3):
    w = str(text).split()
    for i in range(len(w)-mr):
        if " ".join(w[i:i+mr]) in " ".join(w[i+mr:]): return True
    return False

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
model.eval()

class DS(TorchDataset):
    def __init__(self, texts):
        self.enc = tokenizer(texts, max_length=MAX_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
    def __len__(self): return len(self.enc["input_ids"])
    def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}

def run_eval(max_bytes, label):
    inputs, refs = [], []
    for _, row in val_split.iterrows():
        eng = extract_first_sentence(str(row["translation"]))
        akk = tag(truncate(str(row["transliteration"]), max_bytes))
        if eng.strip() and akk.strip():
            inputs.append(PREFIX + akk)
            refs.append(eng)
    logger.info(f"\n=== {label}: {len(inputs)} samples ===")
    loader = DataLoader(DS(inputs), batch_size=4, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label):
            out = model.generate(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device),
                                 max_length=MAX_LENGTH, num_beams=4, early_stopping=True)
            preds.extend([d.strip() for d in tokenizer.batch_decode(out, skip_special_tokens=True)])
    preds_clean = [repeat_cleanup(p) for p in preds]
    for pl, pp in [("raw", preds), ("clean", preds_clean)]:
        chrf = sacrebleu.corpus_chrf(pp, [refs], word_order=2)
        bleu = sacrebleu.corpus_bleu(pp, [refs])
        geo = math.sqrt(chrf.score * bleu.score) if chrf.score > 0 and bleu.score > 0 else 0.0
        rep = 100 * sum(has_repetition(p) for p in pp) / len(pp)
        pred_lens = [len(p.encode('utf-8')) for p in pp]
        logger.info(f"  {pl:5s}: chrF++={chrf.score:.2f}, BLEU={bleu.score:.2f}, geo={geo:.2f}, rep={rep:.1f}%, pred={np.mean(pred_lens):.0f}B")

run_eval(200, "200B truncation")
run_eval(300, "300B truncation")
run_eval(512, "512B truncation (no trunc)")
