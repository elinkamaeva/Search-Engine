import argparse
from indexers import BM25Indexer, Word2VecIndexer, FastTextIndexer, BERTIndexer

# Определение путей к файлам индексов и моделей
BM25_PICKLE = "data/bm25_index.pkl"
W2V_MODEL_PATH = "data/word2vec.bin"
W2V_EMBEDDINGS = "data/w2v_embeddings.npy"
FT_MODEL_PATH = "data/fasttext.model"
FT_EMBEDDINGS = "data/ft_embeddings.npy"
BERT_MODEL_PATH = "ai-forever/sbert_large_nlu_ru"
BERT_EMBEDDINGS = "data/sbert_embeddings.npy"
CORPUS_PATH = "data/corpus.txt"


def main():
    parser = argparse.ArgumentParser(description="Search documents using different indexers.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument("--indexer", type=str, choices=["bm25", "word2vec", "fasttext", "bert"], default="bm25", help="Indexing method.")
    
    args = parser.parse_args()
    
    # Выбор уже инициализированного и загруженного индексатора
    if args.indexer == "bm25":
        selected_indexer = BM25Indexer()
        selected_indexer.load_index(BM25_PICKLE)
    elif args.indexer == "word2vec":
        selected_indexer = Word2VecIndexer()
        selected_indexer.load_index(W2V_EMBEDDINGS, W2V_MODEL_PATH, CORPUS_PATH)
    elif args.indexer == "fasttext":
        selected_indexer = FastTextIndexer()
        selected_indexer.load_index(FT_EMBEDDINGS, FT_MODEL_PATH, CORPUS_PATH)
    elif args.indexer == "bert":
        selected_indexer = BERTIndexer(BERT_MODEL_PATH)
        selected_indexer.load_index(BERT_EMBEDDINGS, CORPUS_PATH)
    
    # Выполнение поиска и вывод результатов
    top_docs = selected_indexer.search(args.query, top_n=3)
    for doc in top_docs:
        print(doc)

if __name__ == "__main__":
    main()
