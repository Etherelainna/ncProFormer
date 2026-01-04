import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from Bio import SeqIO
import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score
)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import _LRScheduler
from collections import Counter
from itertools import product
import torch.optim as optim
import math


torch.cuda.empty_cache()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)

data_dir = Path(".../data")
model_dir = Path(".../gena_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir).to(device)

label_to_id = {'neg': 0, 'pos': 1}
def load_fasta(fasta_file):
    records = list(SeqIO.parse(fasta_file, "fasta"))
    seqs = [str(r.seq) for r in records]
    labels = [label_to_id[r.description.split()[1]] for r in records]
    return seqs, labels

train_seqs, train_labels = load_fasta(data_dir / "train.fasta")
val_seqs, val_labels = load_fasta(data_dir / "val.fasta")
test_seqs, test_labels = load_fasta(data_dir / "test.fasta")


def extract_embeddings(seqs, model, tokenizer, device,
                       batch_size=80, max_length=1024):
    all_mid_tokens = []
    all_cls_vecs   = []
    all_masks      = []

    model.eval()
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            encoded = tokenizer(batch, return_tensors='pt',
                                padding='max_length', truncation=True,
                                max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            input_ids = encoded["input_ids"]

            for h, ids in zip(hidden, input_ids):

                cls_vec = h[0]

                sep_positions = (ids == sep_id).nonzero(as_tuple=False)
                sep_idx = int(sep_positions[0].item()) if len(sep_positions) > 0 else (ids.numel() - 1)

                mid = h[1:sep_idx]
                mask = torch.ones(mid.size(0), dtype=torch.long, device=mid.device)

                all_mid_tokens.append(mid.cpu())
                all_cls_vecs.append(cls_vec.cpu())
                all_masks.append(mask.cpu())

    return all_mid_tokens, all_cls_vecs, all_masks



selected_features = ['ORF.Max.Len', 'Signal.Peak', 'ORF.Max.Cov', 'SNR', 'Seq.pct.Dist','Signal.Min', 'ORF.Integrity', 'Hexamer.Score','Fickett.Score']
train_df = pd.read_csv(".../features_train_build.csv", index_col=0)
val_df = pd.read_csv(".../features_val_build.csv", index_col=0)
test_df = pd.read_csv(".../features_test_build.csv", index_col=0)
train_lnc_feats = train_df[selected_features].values.astype(np.float32)
val_lnc_feats   = val_df[selected_features].values.astype(np.float32)
test_lnc_feats  = test_df[selected_features].values.astype(np.float32)



class RNADataset(Dataset):
    def __init__(self, mids, cls_vecs, masks, feat, labels):
        assert len(mids) == len(labels) == len(cls_vecs) == len(masks) == len(feat)
        self.mids = mids
        self.cls  = cls_vecs
        self.masks = masks
        self.feat = feat
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.mids[idx], self.cls[idx], self.masks[idx], self.feat[idx], self.labels[idx]

def collate_fn(batch):
    mids, cls_vecs, masks, feat, labels = zip(*batch)

    maxL = max(x.size(0) for x in mids)
    H = mids[0].size(1)


    padded_mids = []
    padded_masks = []
    for x, m in zip(mids, masks):
        pad_len = maxL - x.size(0)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, H)], dim=0)
            m = torch.cat([m, torch.zeros(pad_len, dtype=m.dtype)], dim=0)
        padded_mids.append(x)
        padded_masks.append(m)

    mids  = torch.stack(padded_mids, dim=0)
    masks = torch.stack(padded_masks, dim=0)
    cls_v = torch.stack(cls_vecs, dim=0)
    feat  = torch.tensor(feat, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    return mids, cls_v, masks, feat, labels


print("Extracting  embeddings...")
tr_mid, tr_cls, tr_mk = extract_embeddings(train_seqs, model, tokenizer, device)
va_mid, va_cls, va_mk = extract_embeddings(val_seqs,   model, tokenizer, device)
te_mid, te_cls, te_mk = extract_embeddings(test_seqs,  model, tokenizer, device)

train_data = RNADataset(tr_mid, tr_cls, tr_mk, train_lnc_feats, train_labels)
val_data   = RNADataset(va_mid, va_cls, va_mk, val_lnc_feats, val_labels)
test_data  = RNADataset(te_mid, te_cls, te_mk, test_lnc_feats, test_labels)

train_loader = DataLoader(train_data, batch_size=50, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_data,   batch_size=50, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_data,  batch_size=50, shuffle=False, collate_fn=collate_fn)


class HybridConvLinearAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.10, kernel_size=3):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_lin = nn.Linear(embed_dim, embed_dim)
        self.k_lin = nn.Linear(embed_dim, embed_dim)
        self.v_lin = nn.Linear(embed_dim, embed_dim)


        padding = kernel_size // 2

        self.q_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding)
        self.k_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding)
        self.v_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, cls_index: int = 0):

        B, L_all, H = x.size()

        Q_lin = self.q_lin(x)
        K_lin = self.k_lin(x)
        V_lin = self.v_lin(x)

        if attn_mask is None:
            mask_all_bool = torch.ones(B, L_all, device=x.device, dtype=torch.bool)
        else:
            mask_all_bool = (attn_mask != 0)

        mask_all_f = mask_all_bool.to(dtype=x.dtype)

        x_t = x.transpose(1, 2)
        x_t = x_t * mask_all_f.unsqueeze(1)

        Q_conv = self.q_conv(x_t).transpose(1, 2)
        K_conv = self.k_conv(x_t).transpose(1, 2)
        V_conv = self.v_conv(x_t).transpose(1, 2)

        Q_conv = Q_conv * mask_all_f.unsqueeze(-1)
        K_conv = K_conv * mask_all_f.unsqueeze(-1)
        V_conv = V_conv * mask_all_f.unsqueeze(-1)

        Q = Q_conv.clone()
        K = K_conv.clone()
        V = V_conv.clone()
        Q[:, cls_index:cls_index+1, :] = Q_lin[:, cls_index:cls_index+1, :]
        K[:, cls_index:cls_index+1, :] = K_lin[:, cls_index:cls_index+1, :]
        V[:, cls_index:cls_index+1, :] = V_lin[:, cls_index:cls_index+1, :]

        def split_heads(T):
            return T.view(B, L_all, self.num_heads, self.head_dim).transpose(1, 2)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            key_mask = mask_all_bool.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, Vh)
        context = context.transpose(1, 2).contiguous().view(B, L_all, H)

        return self.out_proj(context)


class CNNTransEnc1DClassifier(nn.Module):
    def __init__(self, num_layers=1, embed_dim=768,
                 conv_out_dim=384, out_dim=384,
                 num_heads=3, num_classes=2,
                 dropout=0.10, pooling="meanmax", feat_dim=9):
        super().__init__()
        if conv_out_dim % num_heads != 0:
            raise ValueError(f"conv_out_dim ({conv_out_dim}) must be divisible by num_heads ({num_heads}).")
        self.pooling = pooling

        self.input_proj = nn.Linear(embed_dim, conv_out_dim)
        self.cls_proj   = nn.Linear(embed_dim, conv_out_dim)

        self.attention = HybridConvLinearAttention(conv_out_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(conv_out_dim, conv_out_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conv_out_dim * 2, conv_out_dim)
        )
        self.norm1 = nn.LayerNorm(conv_out_dim)
        self.norm2 = nn.LayerNorm(conv_out_dim)

        self.feat_proj = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 384),
            nn.ReLU()
        )

        pooled_dim = conv_out_dim * 2 + 384 if pooling == "meanmax" else conv_out_dim + 384
        self.fc = nn.Sequential(
            nn.Linear(pooled_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes)
        )

    def masked_mean(self, X, mask):

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        return (X * mask.unsqueeze(-1)).sum(dim=1) / denom.squeeze(-1)

    def masked_max(self, X, mask):
        very_neg = torch.finfo(X.dtype).min
        X_masked = X.masked_fill(mask.unsqueeze(-1) == 0, very_neg)
        return X_masked.max(dim=1).values

    def forward(self, mids, cls_vec, mask, feats):

        B, L, _ = mids.size()

        X_tokens = self.input_proj(mids)
        X_cls    = self.cls_proj(cls_vec)
        X_all    = torch.cat([X_cls.unsqueeze(1), X_tokens], dim=1)

        mask_all = torch.cat([torch.ones(B, 1, device=mask.device, dtype=mask.dtype), mask], dim=1)

        attn_out = self.attention(X_all, attn_mask=mask_all, cls_index=0)
        X_all = self.norm1(X_all + attn_out)
        ffn_out = self.ffn(X_all)
        X_all = self.norm2(X_all + ffn_out)

        if self.pooling == "mean":
            pooled = self.masked_mean(X_all, mask_all)
        elif self.pooling == "max":
            pooled = self.masked_max(X_all, mask_all)
        elif self.pooling == "meanmax":
            mean_pooled = self.masked_mean(X_all, mask_all)
            max_pooled  = self.masked_max(X_all, mask_all)
            pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
        else:
            raise ValueError("pooling must be 'mean', 'max', or 'meanmax'")

        feat_emb = self.feat_proj(feats)  

        combined = torch.cat([pooled, feat_emb], dim=1)

        return self.fc(combined)


epochs = 150
classifier = CNNTransEnc1DClassifier().to(device)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=8e-6, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.8,
    patience=5,
    threshold=1e-5,
    threshold_mode='rel',
    cooldown=0,
    min_lr=4e-6,
    verbose=True
)


print("Start training...")
train_loss_history = []
train_acc_history = []
val_acc_history = []
val_loss_history = []

patience = 10
counter = 0
min_loss = 100
best_epoch = 0
best_acc = 0
early_stop = False

global_step = 0

for epoch in range(epochs):
    classifier.train()
    total_loss = 0
    correct = 0
    for mids, cls_v, masks, feats, labels in train_loader:
        mids, cls_v, masks, feats, labels = mids.to(device), cls_v.to(device), masks.to(device), feats.to(device), labels.to(device)
        logits = classifier(mids, cls_v, masks, feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
    

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")


    classifier.eval()
    val_correct = 0
    val_loss = 0
    with torch.no_grad():
        for mids, cls_v, masks, feats, labels in val_loader:
            mids, cls_v, masks, feats, labels = mids.to(device), cls_v.to(device), masks.to(device), feats.to(device), labels.to(device)
            logits = classifier(mids, cls_v, masks, feats)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            val_correct += (logits.argmax(dim=1) == labels).sum().item()

    val_acc = val_correct / len(val_loader.dataset)
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)
    val_acc_history.append(val_acc)
    scheduler.step(avg_val_loss)

    print(f"val Loss: {val_loss:.4f},avg val Loss: {avg_val_loss:.4f},Val Acc: {val_acc:.4f}")

    if avg_val_loss <= min_loss:
        min_loss = avg_val_loss
        best_epoch = epoch
        counter = 0
        torch.save(classifier.state_dict(), ".../best_classifier_loss.pt")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping !")
            early_stop = True
            break
    


classifier.load_state_dict(torch.load(".../best_classifier_loss.pt"))
classifier.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for mids, cls_v, masks, feats, labels in test_loader:
        mids, cls_v, masks, feats, labels = mids.to(device), cls_v.to(device), masks.to(device), feats.to(device), labels.to(device)
        logits = classifier(mids, cls_v, masks, feats)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())


acc = accuracy_score(all_labels, all_preds)
pre = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
sp = tn / (tn + fp)
auc = roc_auc_score(all_labels, all_probs)

print(f"ACC: {acc*100:.2f}%")
print(f"PRE: {pre*100:.2f}%")
print(f"SN (Recall): {recall*100:.2f}%")
print(f"SP: {sp*100:.2f}%")
print(f"F-score: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")

