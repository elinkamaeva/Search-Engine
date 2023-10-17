import argparse
from indexers import BM25Indexer, VectorIndexer, Word2VecIndexer, FastTextIndexer, BERTIndexer

# Определение путей к моделям и корпусу
CORPUS_PATH = "data/corpus.txt"
BM25_PICKLE = "data/bm25_index.pkl"
W2V_MODEL_PATH = "data/word2vec.bin"
W2V_EMBEDDINGS = "data/w2v_embeddings.npy"
FT_MODEL_PATH = "data/fasttext.model"
FT_EMBEDDINGS = "data/ft_embeddings.npy"
BERT_MODEL_PATH = "ai-forever/sbert_large_nlu_ru"
BERT_EMBEDDINGS = "data/sbert_embeddings.npy"

def main():
    parser = argparse.ArgumentParser(description="Search documents using different indexers.")
    
    # Определение аргументов
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument("--indexer", type=str, choices=["bm25", "word2vec", "fasttext", "bert"], default="bm25", help="Indexing method.")
    
    args = parser.parse_args()
    
    # Выбор индексатора и загрузка индекса из файла
    if args.indexer == "bm25":
        indexer = BM25Indexer()
        indexer.load_index(BM25_PICKLE)
    elif args.indexer == "word2vec":
        indexer = Word2VecIndexer()
        indexer.load_index(W2V_EMBEDDINGS, W2V_MODEL_PATH, CORPUS_PATH)
    elif args.indexer == "fasttext":
        indexer = FastTextIndexer()
        indexer.load_index(FT_EMBEDDINGS, FT_MODEL_PATH, CORPUS_PATH)
    elif args.indexer == "bert":
        indexer = BERTIndexer(BERT_MODEL_PATH)
        indexer.load_index(BERT_EMBEDDINGS, CORPUS_PATH)
    
    # Выполнение поиска и вывод результатов
    top_docs = indexer.search(args.query)
    for doc in top_docs:
        print(doc)

if __name__ == "__main__":
    main()
