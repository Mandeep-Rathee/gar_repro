from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex
import pyterrier as pt



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="biology")
parser.add_argument('--k', type=int, default=16)

args = parser.parse_args()


index = PisaIndex(f"./indexes/pisa/{args.task}")


dataset = pt.get_dataset(f'irds:bright/{args.task}')

# Wrapper to sanitize corpus documents for PISA
def clean_corpus_iter(corpus_iter):
    for doc in corpus_iter:
        # Remove null characters that cause C extension errors
        if 'text' in doc and doc['text'] is not None:
            doc['text'] = doc['text'].replace('\0', '')
        yield doc


print("Creating graph...")

graph = CorpusGraph.from_retriever(
    index.bm25(num_results=args.k+1, k1=0.9, b=0.3), # K+1 needed because retriever will return original document
    clean_corpus_iter(dataset.get_corpus_iter()),
    f"./graphs/pisa/{args.task}.gbm25.{args.k}",
    k=args.k)

print("Graph created")

print(graph)



