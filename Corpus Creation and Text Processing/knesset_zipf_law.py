import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def zipfs_law(input,output):
    df = pd.read_csv(input)
    text = ' '.join(str(sentence) for sentence in df["sentence_text"])
    tokens = text.split()
    tokens=[token for token in tokens if not (token=="."or token=="?" or token=='"' or token=="!")]
    frequency = Counter(tokens)
    sorted_tokens = sorted(frequency, key=frequency.get, reverse=True)
    ranks = range(1, len(sorted_tokens) + 1)
    freqs = [frequency[token] for token in sorted_tokens]
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, freqs, marker="o")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Frequency)")
    plt.title("Zipf's Law")
    plt.grid(True)
    plt.savefig(output)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    zipfs_law(args.input_path,args.output_path)
main()
