import itertools
from collections import Counter
import pandas as pd
from pathlib import Path

class ExtractORF:
    def __init__(self, seq):
        self.seq = seq.upper().replace("U", "T")
        self.result = (0, 0, 0, 0)
        self.longest = 0

    def codons(self, frame):
        start_coord = frame
        while start_coord + 3 <= len(self.seq):
            yield (self.seq[start_coord:start_coord+3], start_coord)
            start_coord += 3

    def longest_orf_in_seq(self, frame_number, start_codons, stop_codons):
        codon_iter = self.codons(frame_number)
        while True:
            try:
                codon, index = next(codon_iter)
            except StopIteration:
                break
            if codon in start_codons:
                ORF_start = index
                integrity = -1
                while True:
                    try:
                        codon, index = next(codon_iter)
                    except StopIteration:
                        break
                    if codon in stop_codons:
                        integrity = 1
                        ORF_end = index + 3
                        ORF_length = ORF_end - ORF_start
                        if ORF_length > self.longest or (
                            ORF_length == self.longest and ORF_start < self.result[1]
                        ):
                            self.longest = ORF_length
                            self.result = (integrity, ORF_start, ORF_end, ORF_length)
                        break

    def longest_ORF(self, start=["ATG"], stop=["TAA", "TAG", "TGA"]):
        for frame in range(3):
            self.longest_orf_in_seq(frame, start, stop)
        ORF_integrity, ORF_start, ORF_end, ORF_length = self.result
        orf_seq = self.seq[ORF_start:ORF_end]
        return ORF_length, ORF_integrity, orf_seq


def fasta_iter(path):
    with open(path, "r") as f:
        sid, seq, label = None, [], None
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(seq), label
                parts = line[1:].split()
                sid = parts[0]
                label = parts[1].lower()
                seq = []
            else:
                seq.append(line)
        if sid is not None:
            yield sid, "".join(seq), label


def build_hexamer_table(fasta_path, out_tsv, k=6, alpha=1.0):
    cod_cnt, non_cnt = Counter(), Counter()
    cod_total, non_total = 0, 0

    for sid, seq, label in fasta_iter(fasta_path):
        extractor = ExtractORF(seq)
        orf_len, integrity, orf_seq = extractor.longest_ORF()
        if orf_len < k:
            continue
        for i in range(len(orf_seq)-k+1):
            kmer = orf_seq[i:i+k]
            if all(c in "ACGT" for c in kmer):
                if label == "pos":
                    cod_cnt.update([kmer])
                    cod_total += 1
                elif label == "neg":
                    non_cnt.update([kmer])
                    non_total += 1

    alphabet = "ACGT"
    all_hex = ["".join(p) for p in itertools.product(alphabet, repeat=k)]
    V = len(all_hex)

    cod_denom = cod_total + alpha * V
    non_denom = non_total + alpha * V
    rows = []
    for kmer in all_hex:
        p_cod = (cod_cnt.get(kmer, 0) + alpha) / cod_denom
        p_non = (non_cnt.get(kmer, 0) + alpha) / non_denom
        rows.append((kmer, p_cod, p_non))

    df = pd.DataFrame(rows, columns=["hexamer", "coding", "noncoding"]).sort_values("hexamer")
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    print(f"Hexamer OK")
 
    return df


fasta_path = ".../data/.../train.fasta"
out_tsv = ".../data/.../RNA_Hexamer.tsv"


df_hex = build_hexamer_table(fasta_path, out_tsv)
df_hex.head()