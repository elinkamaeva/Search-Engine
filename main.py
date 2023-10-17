import argparse
from indexers import bm25_indexer, word2vec_indexer, fasttext_indexer, bert_indexer

def main():
    parser = argparse.ArgumentParser(description="Search documents using different indexers.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument("--indexer", type=str, choices=["bm25", "word2vec", "fasttext", "bert"], default="bm25", help="Indexing method.")
    
    args = parser.parse_args()
    
    # Выбор уже инициализированного и загруженного индексатора
    if args.indexer == "bm25":
        selected_indexer = bm25_indexer
    elif args.indexer == "word2vec":
        selected_indexer = word2vec_indexer
    elif args.indexer == "fasttext":
        selected_indexer = fasttext_indexer
    elif args.indexer == "bert":
        selected_indexer = bert_indexer
    
    # Выполнение поиска и вывод результатов
    top_docs = selected_indexer.search(args.query)
    for doc in top_docs:
        print(doc)

if __name__ == "__main__":
    main()
