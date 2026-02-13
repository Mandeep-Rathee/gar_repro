import os
import pandas as pd
import json
import argparse

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import pyterrier as pt
import pyterrier_alpha as pta
import ir_datasets
from ir_measures import nDCG, R


from gar import GAR
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex

from get_model import GetRanker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="biology")
    parser.add_argument('--model_name', type=str, default="tfrank-0.6b")
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument("--batch", type=int, default=16, help="batch size for reranker")
    parser.add_argument("--graph", type=str, default="gbm25")
    parser.add_argument("--retriever", type=str, default="bm25")
    parser.add_argument("--k", type=int, default=16, help="k for graph based ranking")
    parser.add_argument("--reasoning", action='store_true', help="reasoning")

    args = parser.parse_args()

    graph = CorpusGraph.load(f"./graphs/pisa/{args.task}.gbm25.{args.k}")

    dataset = pt.get_dataset(f'irds:bright/{args.task}')

    ir_dataset = ir_datasets.load(f'bright/{args.task}')

    docstore = ir_dataset.docs_store()


    class TextLoader(pt.Transformer):   
        """Swaps 'query' (gpt4_reason) with 'text' column for re-ranking"""
        def __init__(self, docstore):
            self.docstore = docstore
            file_name = f"./indexes/pisa/stackoverflow/docnos_with_space.json"
            self.docnos_with_space = {}
            if os.path.exists(file_name):
                with open(file_name, 'r') as f:
                    self.docnos_with_space = json.load(f)
            super().__init__()

        def transform(self, df):
            df = df.copy()

            def get_text(docno):
                if docno in self.docnos_with_space:
                    return self.docstore.get(self.docnos_with_space[docno]).text
                return self.docstore.get(docno).text
            
            df['text'] = df['docno'].apply(get_text)
            return df


    index = PisaIndex(f"./indexes/pisa/{args.task}")
    bm25_base = index.bm25(num_results=100, k1=0.9, b=0.4)
    retriever = pt.rewrite.tokenise() >> bm25_base >> pt.rewrite.reset() 

    ranker = GetRanker(model_name=args.model_name, model_type=args.model_type, batch_size=args.batch, task=args.task, reasoning=args.reasoning)

    
    print(f"Using {args.model_name} ranker")



    if args.task == "stackoverflow":
        text_loader = TextLoader(docstore)
        scorer = text_loader >> ranker
    else:
        scorer = pt.text.get_text(dataset, 'text') >> ranker

    

    topics = dataset.get_topics()
    qrels = dataset.get_qrels()

    topics = topics[['qid', 'text']].rename(columns={'text': 'query'})


    save_dir = f"./saved_runs/{args.graph}/{args.retriever}/{args.task}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', None)  # Display full width of the terminal


    results= pt.Experiment(
        [retriever % args.budget, 
        retriever % args.budget >> scorer,
        retriever % args.budget >> GAR(scorer, graph, batch_size=args.batch, num_results=args.budget, verbose=True),
        ],
        topics,
        qrels,
        [nDCG@10, nDCG@args.budget, R@10, R@args.budget],
        names=[f'{args.retriever}.c{args.budget}', 
        f'{args.retriever}_{args.model_name}.c{args.budget}',
        f'{args.retriever}_GAR({args.model_name}).c{args.budget}',
        ],
        save_dir=save_dir,
        save_mode = 'reuse'
    )
    print(f"Task: {args.task}")    
    print(results.T)

if __name__ == '__main__':
    main()
