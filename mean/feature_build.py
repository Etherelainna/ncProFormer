import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def fasta_iter(path):
    with open(path, "r", encoding="utf-8") as f:
        sid, buf = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(buf)

                sid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if sid is not None:
            yield sid, "".join(buf)

def fasta_to_map(path):
    m = {}
    for sid, seq in fasta_iter(path):
        m[sid] = seq
    return m

class ExtractORF:
    def __init__(self, seq):
        self.seq = seq.upper().replace("U", "T")
        self.result = (0, 0, 0, 0)
        self.longest = 0

    def codons(self, frame):
        pos = frame
        while pos + 3 <= len(self.seq):
            yield self.seq[pos:pos+3], pos
            pos += 3

    def longest_orf_in_frame(self, frame, start_codons, stop_codons):
        it = self.codons(frame)
        for codon, idx in it:
            if codon in start_codons:
                orf_start = idx
                integrity = 0

                for codon2, idx2 in it:
                    if codon2 in stop_codons:
                        integrity = 1
                        orf_end = idx2 + 3
                        length = orf_end - orf_start
                        if length > self.longest or (length == self.longest and orf_start < self.result[1]):
                            self.longest = length
                            self.result = (integrity, orf_start, orf_end, length)
                        break


    def longest_ORF(self, start=("ATG",), stop=("TAA", "TAG", "TGA")):
        for frame in range(3):
            self.longest_orf_in_frame(frame, start, stop)
        integ, s, e, L = self.result
        orf_seq = self.seq[s:e] if L > 0 else ""
        return L, integ, orf_seq


def load_hex_table(tsv_path):
    t = pd.read_csv(tsv_path, sep="\t")
    cod = dict(zip(t["hexamer"], t["coding"]))
    non = dict(zip(t["hexamer"], t["noncoding"]))
    return cod, non

def hexamer_score_for_orf(orf_seq, cod_freq, non_freq, k=6):
    if not orf_seq or len(orf_seq) < k:
        return 0.0
    s = orf_seq.upper().replace("U", "T")
    logs = []
    for i in range(len(s) - k + 1):
        kmer = s[i:i+k]
        if set(kmer) <= {"A", "C", "G", "T"}:
            p = cod_freq.get(kmer, 1e-12)
            q = non_freq.get(kmer, 1e-12)
            logs.append(math.log(p / q))
    return float(sum(logs) / len(logs)) if logs else 0.0


position_prob = {
    "A": [0.94, 0.68, 0.84, 0.93, 0.58, 0.68, 0.45, 0.34, 0.20, 0.22],
    "C": [0.80, 0.70, 0.70, 0.81, 0.66, 0.48, 0.51, 0.33, 0.30, 0.23],
    "G": [0.90, 0.88, 0.74, 0.64, 0.53, 0.48, 0.27, 0.16, 0.08, 0.08],
    "T": [0.97, 0.97, 0.91, 0.68, 0.69, 0.44, 0.54, 0.20, 0.09, 0.09],
}
position_weight = {"A": 0.26, "C": 0.18, "G": 0.31, "T": 0.33}
position_para = [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 0.0]

content_prob = {
    "A": [0.28, 0.49, 0.44, 0.55, 0.62, 0.49, 0.67, 0.65, 0.81, 0.21],
    "C": [0.82, 0.64, 0.51, 0.64, 0.59, 0.59, 0.43, 0.44, 0.39, 0.31],
    "G": [0.40, 0.54, 0.47, 0.64, 0.64, 0.73, 0.41, 0.41, 0.33, 0.29],
    "T": [0.28, 0.24, 0.39, 0.40, 0.55, 0.75, 0.56, 0.69, 0.51, 0.58],
}
content_weight = {"A": 0.11, "C": 0.12, "G": 0.15, "T": 0.14}
content_para = [0.33, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.17, 0.0]

def _lookup(value, base, paras, probs, weight):
    if float(value) < 0:
        return 0.0
    for idx, val in enumerate(paras):
        if float(value) >= val:
            return float(probs[base][idx]) * float(weight[base])
    return 0.0

def fickett_value(seq):
    s = seq.upper().replace("U", "T")
    s = "".join([c for c in s if c in "ACGT"])
    if len(s) < 2:
        return 0.0

    n = len(s)
    A_content = s.count("A") / n
    C_content = s.count("C") / n
    G_content = s.count("G") / n
    T_content = s.count("T") / n

    p0 = s[0::3]; p1 = s[1::3]; p2 = s[2::3]
    def pos_ratio(b):
        maxv = max(p0.count(b), p1.count(b), p2.count(b))
        minv = min(p0.count(b), p1.count(b), p2.count(b))
        return maxv / (minv + 1.0)

    A_pos = pos_ratio("A"); C_pos = pos_ratio("C")
    G_pos = pos_ratio("G"); T_pos = pos_ratio("T")

    score = 0.0
    score += _lookup(A_content, "A", content_para, content_prob, content_weight)
    score += _lookup(C_content, "C", content_para, content_prob, content_weight)
    score += _lookup(G_content, "G", content_para, content_prob, content_weight)
    score += _lookup(T_content, "T", content_para, content_prob, content_weight)

    score += _lookup(A_pos, "A", position_para, position_prob, position_weight)
    score += _lookup(C_pos, "C", position_para, position_prob, position_weight)
    score += _lookup(G_pos, "G", position_para, position_prob, position_weight)
    score += _lookup(T_pos, "T", position_para, position_prob, position_weight)

    return float(score)


def main(args):
    csv_in = Path(args.lncfinder_csv)
    fasta = Path(args.fasta)
    hex_tsv = Path(args.hexamer_tsv)

    df = pd.read_csv(csv_in, index_col=0)
    df.index.name = "ID"
    df = df.reset_index()

    seq_map = fasta_to_map(fasta)

    cod_freq, non_freq = load_hex_table(hex_tsv)

    orf_integrities = []
    hex_scores = []
    fickett_scores = []

    miss = 0
    for _id in df["ID"].astype(str):
        seq = seq_map.get(_id, "")
        if not seq:
            miss += 1
            orf_integrities.append(0)
            hex_scores.append(0.0)
            fickett_scores.append(0.0)
            continue


        extractor = ExtractORF(seq)
        orf_len, integrity, orf_seq = extractor.longest_ORF()
        orf_integrities.append(int(integrity))


        hs = hexamer_score_for_orf(orf_seq, cod_freq, non_freq, k=6)
        hex_scores.append(hs)


        fs = fickett_value(seq)
        fickett_scores.append(fs)

    if miss:
        print(f"warning: {miss} ID were not find")


    df["ORF.Integrity"] = orf_integrities
    df["Hexamer.Score"] = hex_scores
    df["Fickett.Score"] = fickett_scores


    out_path = csv_in.with_name(csv_in.stem + "_build.csv")
    df.to_csv(out_path, index=False)
    print("OK：", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add ORF integrity, Hexamer Score, Fickett Score to LncFinder CSV.")
    parser.add_argument("--lnc", required=True, help="Path to LncFinder features CSV.")
    parser.add_argument("--fasta", required=True, help="Path to fasta containing sequences with IDs.")
    parser.add_argument("--hexamer", required=True, help="Path to ncRNA_Human_Hexamer.tsv (hexamer, coding, noncoding).")
    args = parser.parse_args()
    main(args)



