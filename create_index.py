import ir_datasets
import pyterrier as pt
from pyterrier_pisa import PisaIndex
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="biology")
args = parser.parse_args()


print("Indexing...")

pt_dataset = pt.get_dataset(f'irds:bright/{args.task}')



# save ids with space in a file
file_name = f"./indexes/pisa/{args.task}/docnos_with_space.json"

docs_with_space = {}



for doc in pt_dataset.get_corpus_iter():
    if ' ' in doc['docno']:
        print(doc['docno'])
        docs_with_space[doc['docno'].replace(' ', '_')] =  doc['docno']


with open(file_name, "w") as f:
    json.dump(docs_with_space, f)
    f.close()



# Wrapper to sanitize corpus documents for PISA
def clean_corpus_for_pisa(corpus_iter):
    for doc in corpus_iter:
        # Remove null characters that cause C extension errors
        if 'text' in doc and doc['text'] is not None:
            doc['text'] = doc['text'].replace('\0', '')
        # PISA doesn't handle spaces well in docnos - replace with underscores
        if 'docno' in doc:
            doc['docno'] = doc['docno'].replace(' ', '_')
        yield doc

index_path = f"./indexes/pisa/{args.task}"
index = PisaIndex(index_path)
index.index(clean_corpus_for_pisa(pt_dataset.get_corpus_iter()))
print("Index reference: ", index)



print("Indexing complete")