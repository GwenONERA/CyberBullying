#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
CyberBullyingAdo — 3-class Hate-Speech Classification Pipeline
Encoder : Yanzhu/bertweetfr-base  (CamemBERT-based, French tweets)
Labels  : CAG (covert) · NAG (non-aggressive) · OAG (overt)

Phase 1 : frozen embeddings + Optuna-tuned classifier            (fast)
Phase 2 : end-to-end fine-tuning + Optuna                        (deep)
Final   : best approach retrained on train+val, 5-seed test eval
═══════════════════════════════════════════════════════════════════════════
"""

import os, sys, random, warnings, gc, time, contextlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, normalize as sk_normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SEED       = 10
MODEL_NAME = "Yanzhu/bertweetfr-base"
P1_TRIALS  = 200       # frozen-embedding Optuna trials
P2_TRIALS  = 40        # fine-tuning Optuna trials
EVAL_SEEDS = [42, 123, 456, 789, 1024]
EMB_BATCH  = 128        # batch size for embedding extraction
NUM_LABELS = 3

LAYER_KEYS = ["last", "last-1", "last-2", "last-3", "mean_last4", "concat_last4"]
POOL_KEYS  = ["cls", "mean", "max"]

def layer_indices(key):
    mapping = {
        "last": [-1], "last-1": [-2], "last-2": [-3], "last-3": [-4],
        "mean_last4": [-4, -3, -2, -1], "concat_last4": [-4, -3, -2, -1],
    }
    return mapping[key]


def seed_everything(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print(f"[config] device={DEVICE}  amp={USE_AMP}  P1_trials={P1_TRIALS}  P2_trials={P2_TRIALS}")

# AMP helpers compatible with torch ≥ 1.13
def amp_autocast():
    if USE_AMP:
        return torch.cuda.amp.autocast()
    return contextlib.nullcontext()

def amp_scaler():
    if USE_AMP:
        return torch.cuda.amp.GradScaler()
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  §1  DATA LOADING & TOKEN-LENGTH FILTERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  §1  DATA LOADING & FILTERING")
print("="*72)

df = pd.read_parquet("./CyberBullyingAdo.parquet")
print(f"Raw dataset: {len(df)} rows")
print(f"Class distribution (raw):\n{df['HATE'].value_counts().to_string()}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL_MAX_LEN = min(getattr(tokenizer, "model_max_length", 512), 512)
print(f"Tokenizer max length cap: {MODEL_MAX_LEN}")

# Count tokens per example
def token_count(text: str) -> int:
    return len(tokenizer.encode(str(text), add_special_tokens=True))

df["TEXT"] = df["TEXT"].astype(str)
df["n_tokens"] = df["TEXT"].apply(token_count)

before = len(df)
mask = (df["n_tokens"] >= 2) & (df["n_tokens"] <= MODEL_MAX_LEN)
df = df[mask].reset_index(drop=True)
after = len(df)
print(f"Filtered: {before} → {after}  (removed {before - after})")
print(f"Token stats: min={df['n_tokens'].min()}  max={df['n_tokens'].max()}"
      f"  mean={df['n_tokens'].mean():.1f}  median={df['n_tokens'].median():.0f}")
print(f"Class distribution (filtered):\n{df['HATE'].value_counts().to_string()}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  §2  LABEL ENCODING & STRATIFIED SPLITS  (60 / 20 / 20)
# ═══════════════════════════════════════════════════════════════════════════
print("="*72)
print("  §2  STRATIFIED SPLITS")
print("="*72)

le = LabelEncoder()
le.fit(sorted(df["HATE"].unique()))          # deterministic order: CAG, NAG, OAG
df["label"] = le.transform(df["HATE"])
label_names = list(le.classes_)
print(f"Label encoding: {dict(zip(label_names, le.transform(label_names)))}")

texts_all  = df["TEXT"].values
labels_all = df["label"].values

# Split 1: 60 % train vs 40 % temp
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
idx_train, idx_temp = next(sss1.split(texts_all, labels_all))

# Split 2: 50 % of temp → 20 % val + 20 % test
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
idx_val, idx_test = next(sss2.split(texts_all[idx_temp], labels_all[idx_temp]))
idx_val  = idx_temp[idx_val]
idx_test = idx_temp[idx_test]

texts_train,  y_train  = texts_all[idx_train],  labels_all[idx_train]
texts_val,    y_val     = texts_all[idx_val],    labels_all[idx_val]
texts_test,   y_test    = texts_all[idx_test],   labels_all[idx_test]

for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = {label_names[c]: int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    print(f"  {name:5s}  n={len(y):5d}  {dist}")
print()

# Class weights
cw_np = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=y_train)
CLASS_WEIGHTS = torch.tensor(cw_np, dtype=torch.float32, device=DEVICE)
print(f"Class weights: {dict(zip(label_names, cw_np.round(4)))}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  §3  EMBEDDING EXTRACTION  (frozen encoder, all layer/pool combos)
# ═══════════════════════════════════════════════════════════════════════════
print("="*72)
print("  §3  EMBEDDING EXTRACTION")
print("="*72)

def pool_hidden(hidden, attention_mask, pool_type):
    """Pool token-level hidden states → sentence embedding."""
    if pool_type == "cls":
        return hidden[:, 0, :]                          # first token (<s>)
    mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
    if pool_type == "mean":
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    if pool_type == "max":
        hidden = hidden.masked_fill(mask == 0, -1e9)
        return hidden.max(dim=1).values
    raise ValueError(pool_type)


def aggregate_layers(hidden_states, layer_key):
    """Select / aggregate layers from hidden_states tuple."""
    idxs = layer_indices(layer_key)
    selected = [hidden_states[i] for i in idxs]
    if layer_key == "concat_last4":
        return torch.cat(selected, dim=-1)              # (B, L, 4*H)
    if layer_key == "mean_last4":
        return torch.stack(selected, dim=0).mean(0)     # (B, L, H)
    return selected[0]                                  # single layer


def extract_embeddings(texts_list, batch_size=EMB_BATCH):
    """
    One forward pass per batch → collect all (layer, pool) embeddings.
    Returns dict[(layer_key, pool_key)] → np.ndarray (N, dim).
    """
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    accum = defaultdict(list)
    n = len(texts_list)
    t0 = time.time()
    for i in range(0, n, batch_size):
        batch_texts = list(texts_list[i:i+batch_size])
        inputs = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=MODEL_MAX_LEN, return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad(), amp_autocast():
            out = encoder(**inputs, output_hidden_states=True, return_dict=True)
        hs  = out.hidden_states                         # (embed, l1, …, l12)
        attn = inputs["attention_mask"]
        for lk in LAYER_KEYS:
            agg = aggregate_layers(hs, lk)              # (B, L, D)
            for pk in POOL_KEYS:
                emb = pool_hidden(agg, attn, pk)        # (B, D)
                accum[(lk, pk)].append(emb.float().cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"    batch {i//batch_size+1}/{(n-1)//batch_size+1}"
                  f"  ({time.time()-t0:.0f}s)")
    del encoder; gc.collect(); torch.cuda.empty_cache() if USE_AMP else None
    result = {}
    for key in accum:
        result[key] = np.vstack(accum[key])
    print(f"  Extraction done in {time.time()-t0:.0f}s  "
          f"({len(result)} combos, first shape={next(iter(result.values())).shape})")
    return result

print("Extracting TRAIN embeddings …")
emb_train = extract_embeddings(texts_train)
print("Extracting VAL embeddings …")
emb_val   = extract_embeddings(texts_val)
print("Extracting TEST embeddings …")
emb_test  = extract_embeddings(texts_test)

# Show dims
for k in sorted(emb_train.keys())[:3]:
    print(f"  {k}: train={emb_train[k].shape}  val={emb_val[k].shape}")
print()


# ═══════════════════════════════════════════════════════════════════════════
#  §4  PHASE 1 — FROZEN EMBEDDINGS + OPTUNA
# ═══════════════════════════════════════════════════════════════════════════
print("="*72)
print("  §4  PHASE 1 — FROZEN-EMBEDDING CLASSIFIER SEARCH")
print("="*72)


# ── MLP classifier on embeddings ──────────────────────────────────────────
class EmbMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, dropout, n_classes):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_on_embeddings(
    X_tr, y_tr, X_vl, y_vl,
    hidden_dim=256, n_layers=1, dropout=0.3,
    lr=1e-3, weight_decay=1e-4, batch_size=64,
    epochs=60, patience=7, class_weights=None,
):
    """Train small MLP on pre-extracted embeddings.  Returns best val macro-F1."""
    seed_everything(SEED)
    in_dim = X_tr.shape[1]
    model = EmbMLP(in_dim, hidden_dim, n_layers, dropout, NUM_LABELS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights if class_weights is not None else CLASS_WEIGHTS
    )

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)
    yv_np = y_vl

    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_f1, wait = 0.0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(Xv).argmax(1).cpu().numpy()
        f1 = f1_score(yv_np, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    del model; gc.collect()
    return best_f1


def phase1_objective(trial):
    lk   = trial.suggest_categorical("layer",   LAYER_KEYS)
    pk   = trial.suggest_categorical("pool",    POOL_KEYS)
    norm = trial.suggest_categorical("normalize", [True, False])
    clf  = trial.suggest_categorical("clf",     ["logreg", "mlp"])

    X_tr = emb_train[(lk, pk)].copy()
    X_vl = emb_val[(lk, pk)].copy()
    if norm:
        X_tr = sk_normalize(X_tr); X_vl = sk_normalize(X_vl)

    if clf == "logreg":
        C = trial.suggest_float("logreg_C", 1e-4, 1e2, log=True)
        mi = trial.suggest_int("logreg_maxiter", 300, 3000)
        m = LogisticRegression(
            C=C, max_iter=mi, class_weight="balanced",
            solver="lbfgs", multi_class="multinomial", random_state=SEED,
        )
        m.fit(X_tr, y_train)
        preds = m.predict(X_vl)
        return f1_score(y_val, preds, average="macro")
    else:
        hd  = trial.suggest_categorical("mlp_hidden",   [128, 256, 512, 768])
        nl  = trial.suggest_int("mlp_n_layers", 1, 3)
        dr  = trial.suggest_float("mlp_dropout",  0.0, 0.5)
        lr  = trial.suggest_float("mlp_lr",       1e-5, 5e-3, log=True)
        wd  = trial.suggest_float("mlp_wd",       1e-8, 1e-2, log=True)
        bs  = trial.suggest_categorical("mlp_bs", [32, 64, 128])
        return train_mlp_on_embeddings(
            X_tr, y_train, X_vl, y_val,
            hidden_dim=hd, n_layers=nl, dropout=dr,
            lr=lr, weight_decay=wd, batch_size=bs,
        )


t0 = time.time()
study1 = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="phase1_frozen",
)
study1.optimize(phase1_objective, n_trials=P1_TRIALS, show_progress_bar=True)
p1_time = time.time() - t0

print(f"\n  Phase 1 best val macro-F1 = {study1.best_value:.4f}  ({p1_time:.0f}s)")
print(f"  Best params: {study1.best_params}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  §5  PHASE 2 — END-TO-END FINE-TUNING + OPTUNA
# ═══════════════════════════════════════════════════════════════════════════
print("="*72)
print("  §5  PHASE 2 — FINE-TUNING SEARCH")
print("="*72)


class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = list(texts)
        self.labels = list(labels)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts), padding=True, truncation=True,
        max_length=MODEL_MAX_LEN, return_tensors="pt",
    )
    return enc, torch.tensor(labels, dtype=torch.long)


class BertClassifier(nn.Module):
    def __init__(self, pool="mean", layer_key="last",
                 hidden_dim=256, n_layers=1, dropout=0.3, n_classes=NUM_LABELS):
        super().__init__()
        self.pool_type = pool
        self.layer_key = layer_key

        self.enc = AutoModel.from_pretrained(MODEL_NAME)
        H = self.enc.config.hidden_size                 # 768
        feat_dim = H * 4 if layer_key == "concat_last4" else H

        # optional projection for concat
        self.proj = nn.Linear(feat_dim, H) if layer_key == "concat_last4" else nn.Identity()
        in_dim = H

        # attention pooling head (small)
        if pool == "attention":
            self.attn_w = nn.Linear(in_dim, 1)
        else:
            self.attn_w = None

        # classifier head
        layers = []
        prev = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*layers)

    # ── freezing ──────────────────────────────────────────────
    def freeze(self, strategy="full"):
        """strategy: 'none' | 'last_2' | 'last_4' | 'full'"""
        if strategy == "full":
            return                                       # train everything
        for p in self.enc.parameters():
            p.requires_grad = False
        if strategy == "none":
            return
        n_unfreeze = int(strategy.split("_")[1])
        encoder_layers = self.enc.encoder.layer
        for layer in encoder_layers[-n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

    # ── forward ───────────────────────────────────────────────
    def forward(self, input_ids, attention_mask, **kw):
        out = self.enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        agg = aggregate_layers(hs, self.layer_key)      # (B, L, D)
        agg = self.proj(agg)                             # project if concat

        if self.attn_w is not None:                      # attention pooling
            scores = self.attn_w(agg).squeeze(-1)        # (B, L)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (agg * weights).sum(dim=1)
        else:
            pooled = pool_hidden(agg, attention_mask, self.pool_type)

        return self.head(pooled)


def finetune_run(
    texts_tr, y_tr, texts_vl, y_vl,
    pool="mean", layer_key="last", freeze_strategy="full",
    hidden_dim=256, n_layers_head=1, dropout=0.3,
    lr_head=2e-4, lr_enc=2e-5, weight_decay=1e-2,
    batch_size=16, epochs=10, patience=4,
    warmup_ratio=0.06, label_smoothing=0.0,
    report_fn=None,               # optuna trial.report for pruning
    return_state=False,
):
    """
    Fine-tune BertClassifier.
    Returns (best_val_macro_f1, best_state_dict_or_None).
    """
    model = BertClassifier(
        pool=pool, layer_key=layer_key,
        hidden_dim=hidden_dim, n_layers=n_layers_head, dropout=dropout,
    ).to(DEVICE)
    model.freeze(freeze_strategy)

    ds_tr = TweetDataset(texts_tr, y_tr)
    ds_vl = TweetDataset(texts_vl, y_vl)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       collate_fn=collate_fn, drop_last=False, num_workers=0)
    dl_vl = DataLoader(ds_vl, batch_size=batch_size*2, shuffle=False,
                       collate_fn=collate_fn, num_workers=0)

    # separate param groups
    enc_params  = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("enc.")]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("enc.")]
    optimizer = torch.optim.AdamW([
        {"params": enc_params,  "lr": lr_enc},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)

    total_steps = len(dl_tr) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS,
                                  label_smoothing=label_smoothing)
    scaler = amp_scaler()

    best_f1, best_state, wait = 0.0, None, 0

    for ep in range(epochs):
        # ── train ──
        model.train()
        for enc_batch, yb in dl_tr:
            enc_batch = {k: v.to(DEVICE) for k, v in enc_batch.items()}
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = model(**enc_batch)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(**enc_batch)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # ── validate ──
        model.eval()
        all_preds = []
        with torch.no_grad():
            for enc_batch, _ in dl_vl:
                enc_batch = {k: v.to(DEVICE) for k, v in enc_batch.items()}
                with amp_autocast():
                    logits = model(**enc_batch)
                all_preds.append(logits.argmax(1).cpu().numpy())
        preds = np.concatenate(all_preds)
        f1 = f1_score(y_vl, preds, average="macro")

        if report_fn is not None:
            report_fn(f1, ep)

        if f1 > best_f1:
            best_f1, wait = f1, 0
            if return_state:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    del model, optimizer, scheduler; gc.collect()
    torch.cuda.empty_cache() if USE_AMP else None
    return best_f1, best_state


def phase2_objective(trial):
    pool = trial.suggest_categorical("pool",
                ["cls", "mean", "max", "attention"])
    lk   = trial.suggest_categorical("layer", LAYER_KEYS)
    freeze = trial.suggest_categorical("freeze",
                ["none", "last_2", "last_4", "full"])
    hd   = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    nlh  = trial.suggest_int("n_layers_head", 1, 2)
    dr   = trial.suggest_float("dropout", 0.1, 0.5)
    lr_h = trial.suggest_float("lr_head", 5e-5, 5e-3, log=True)
    lr_e = trial.suggest_float("lr_enc",  1e-6, 5e-5, log=True)
    wd   = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    bs   = trial.suggest_categorical("batch_size", [32,64])
    ep   = trial.suggest_int("epochs", 3, 10)
    wu   = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    ls   = trial.suggest_float("label_smoothing", 0.0, 0.15)

    try:
        f1, _ = finetune_run(
            texts_train, y_train, texts_val, y_val,
            pool=pool, layer_key=lk, freeze_strategy=freeze,
            hidden_dim=hd, n_layers_head=nlh, dropout=dr,
            lr_head=lr_h, lr_enc=lr_e, weight_decay=wd,
            batch_size=bs, epochs=ep, patience=4,
            warmup_ratio=wu, label_smoothing=ls,
            report_fn=trial.report, return_state=False,
        )
        if trial.should_prune():
            raise optuna.TrialPruned()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect(); torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        raise
    return f1


t0 = time.time()
study2 = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    study_name="phase2_finetune",
)
study2.optimize(phase2_objective, n_trials=P2_TRIALS, show_progress_bar=True)
p2_time = time.time() - t0

print(f"\n  Phase 2 best val macro-F1 = {study2.best_value:.4f}  ({p2_time:.0f}s)")
print(f"  Best params: {study2.best_params}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  §6  FINAL EVALUATION — MULTI-SEED ON HELD-OUT TEST
# ═══════════════════════════════════════════════════════════════════════════
print("="*72)
print("  §6  FINAL EVALUATION ON HELD-OUT TEST SET")
print("="*72)

best_p1 = study1.best_value
best_p2 = study2.best_value
USE_FINETUNE = best_p2 >= best_p1
print(f"  Phase 1 best = {best_p1:.4f}  |  Phase 2 best = {best_p2:.4f}")
print(f"  → Using {'FINE-TUNE (Phase 2)' if USE_FINETUNE else 'FROZEN (Phase 1)'}\n")

# Merge train + val
texts_trainval = np.concatenate([texts_train, texts_val])
y_trainval     = np.concatenate([y_train, y_val])

results = []           # list of (seed, macro_f1, preds)

for run_seed in EVAL_SEEDS:
    seed_everything(run_seed)
    print(f"  Seed {run_seed} …", end=" ", flush=True)

    if USE_FINETUNE:
        bp = study2.best_params
        # Create a small internal val split from trainval for early stopping
        sss_int = StratifiedShuffleSplit(n_splits=1, test_size=0.10,
                                         random_state=run_seed)
        i_tr, i_vl = next(sss_int.split(texts_trainval, y_trainval))
        f1_val, state = finetune_run(
            texts_trainval[i_tr], y_trainval[i_tr],
            texts_trainval[i_vl], y_trainval[i_vl],
            pool=bp["pool"], layer_key=bp["layer"],
            freeze_strategy=bp["freeze"],
            hidden_dim=bp["hidden_dim"], n_layers_head=bp["n_layers_head"],
            dropout=bp["dropout"],
            lr_head=bp["lr_head"], lr_enc=bp["lr_enc"],
            weight_decay=bp["weight_decay"],
            batch_size=bp["batch_size"],
            epochs=bp["epochs"], patience=5,
            warmup_ratio=bp["warmup_ratio"],
            label_smoothing=bp["label_smoothing"],
            return_state=True,
        )
        # Evaluate on test
        model = BertClassifier(
            pool=bp["pool"], layer_key=bp["layer"],
            hidden_dim=bp["hidden_dim"], n_layers=bp["n_layers_head"],
            dropout=0.0,   # no dropout at eval
        ).to(DEVICE)
        model.load_state_dict(state)
        model.eval()

        ds_te = TweetDataset(texts_test, y_test)
        dl_te = DataLoader(ds_te, batch_size=64, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
        all_preds = []
        with torch.no_grad():
            for enc_batch, _ in dl_te:
                enc_batch = {k: v.to(DEVICE) for k, v in enc_batch.items()}
                with amp_autocast():
                    logits = model(**enc_batch)
                all_preds.append(logits.argmax(1).cpu().numpy())
        preds = np.concatenate(all_preds)
        del model, state; gc.collect()
        torch.cuda.empty_cache() if USE_AMP else None

    else:   # frozen embeddings
        bp = study1.best_params
        lk, pk = bp["layer"], bp["pool"]
        do_norm = bp["normalize"]
        X_tr = np.vstack([emb_train[(lk, pk)], emb_val[(lk, pk)]])
        X_te = emb_test[(lk, pk)].copy()
        if do_norm:
            X_tr = sk_normalize(X_tr); X_te = sk_normalize(X_te)

        if bp["clf"] == "logreg":
            m = LogisticRegression(
                C=bp["logreg_C"], max_iter=bp["logreg_maxiter"],
                class_weight="balanced", solver="lbfgs",
                multi_class="multinomial", random_state=run_seed,
            )
            m.fit(X_tr, y_trainval)
            preds = m.predict(X_te)
        else:
            # MLP: use internal val split for early stopping
            sss_int = StratifiedShuffleSplit(n_splits=1, test_size=0.10,
                                             random_state=run_seed)
            i_tr, i_vl = next(sss_int.split(X_tr, y_trainval))

            in_dim = X_tr.shape[1]
            mlp = EmbMLP(in_dim, bp["mlp_hidden"], bp["mlp_n_layers"],
                         bp["mlp_dropout"], NUM_LABELS).to(DEVICE)
            opt = torch.optim.AdamW(mlp.parameters(), lr=bp["mlp_lr"],
                                     weight_decay=bp["mlp_wd"])
            loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

            ds = TensorDataset(
                torch.tensor(X_tr[i_tr], dtype=torch.float32),
                torch.tensor(y_trainval[i_tr], dtype=torch.long),
            )
            dl = DataLoader(ds, batch_size=bp["mlp_bs"], shuffle=True)
            Xv_t = torch.tensor(X_tr[i_vl], dtype=torch.float32).to(DEVICE)
            yv_np = y_trainval[i_vl]

            best_f1_int, wait, best_st = 0, 0, None
            for ep in range(80):
                mlp.train()
                for xb, yb in dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss_fn(mlp(xb), yb).backward()
                    nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                    opt.step()
                mlp.eval()
                with torch.no_grad():
                    p = mlp(Xv_t).argmax(1).cpu().numpy()
                f1 = f1_score(yv_np, p, average="macro")
                if f1 > best_f1_int:
                    best_f1_int, wait = f1, 0
                    best_st = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                else:
                    wait += 1
                    if wait >= 10:
                        break
            mlp.load_state_dict(best_st)
            mlp.eval()
            Xte_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                preds = mlp(Xte_t).argmax(1).cpu().numpy()
            del mlp; gc.collect()

    test_f1 = f1_score(y_test, preds, average="macro")
    results.append((run_seed, test_f1, preds))
    print(f"macro-F1 = {test_f1:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  §7  REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  §7  FINAL REPORT")
print("="*72)

f1_scores = [r[1] for r in results]
mean_f1   = np.mean(f1_scores)
std_f1    = np.std(f1_scores)
best_idx  = int(np.argmax(f1_scores))
best_seed, best_f1, best_preds = results[best_idx]

print(f"\n  Approach   : {'Fine-tune' if USE_FINETUNE else 'Frozen embeddings'}")
print(f"  Seeds      : {EVAL_SEEDS}")
print(f"  Macro-F1s  : {[f'{f:.4f}' for f in f1_scores]}")
print(f"  Mean±Std   : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Best seed  : {best_seed}  (F1 = {best_f1:.4f})")

# Detailed metrics for best run
print(f"\n  ── Best-run detailed metrics (seed={best_seed}) ──")
print(f"  Accuracy     : {accuracy_score(y_test, best_preds):.4f}")
print(f"  Macro F1     : {best_f1:.4f}")
print(f"  Weighted F1  : {f1_score(y_test, best_preds, average='weighted'):.4f}")

print(f"\n  Classification Report:")
print(classification_report(y_test, best_preds, target_names=label_names, digits=4))

print("  Confusion Matrix (rows=true, cols=pred):")
cm = confusion_matrix(y_test, best_preds)
header = "       " + "  ".join(f"{n:>5s}" for n in label_names)
print(header)
for i, row in enumerate(cm):
    vals = "  ".join(f"{v:5d}" for v in row)
    print(f"  {label_names[i]:>5s}  {vals}")

# Per-class metrics for each seed
print(f"\n  ── Per-class F1 across seeds ──")
print(f"  {'Seed':>6s}", end="")
for n in label_names:
    print(f"  {n:>8s}", end="")
print(f"  {'macro':>8s}")
for seed, f1_all, preds in results:
    per_class = f1_score(y_test, preds, average=None)
    print(f"  {seed:6d}", end="")
    for v in per_class:
        print(f"  {v:8.4f}", end="")
    print(f"  {f1_all:8.4f}")

# Summary
print(f"\n{'='*72}")
if USE_FINETUNE:
    print(f"  Best Phase 2 params: {study2.best_params}")
else:
    print(f"  Best Phase 1 params: {study1.best_params}")
print(f"\n  ★  TEST MACRO-F1 = {mean_f1:.4f} ± {std_f1:.4f}  "
      f"(best single = {best_f1:.4f})")
print(f"{'='*72}\n")
