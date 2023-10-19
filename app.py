from flask import Flask, render_template, request, redirect, url_for
from indexers import BM25Indexer, Word2VecIndexer, FastTextIndexer, BERTIndexer
import time

app = Flask(__name__)

# Определение путей к файлам индексов и моделей
BM25_PICKLE = "data/bm25_index.pkl"
W2V_MODEL_PATH = "data/word2vec.bin"
W2V_EMBEDDINGS = "data/w2v_embeddings.npy"
FT_MODEL_PATH = "data/fasttext.model"
FT_EMBEDDINGS = "data/ft_embeddings.npy"
BERT_MODEL_PATH = "ai-forever/sbert_large_nlu_ru"
BERT_EMBEDDINGS = "data/sbert_embeddings.npy"
CORPUS_PATH = "data/corpus.txt"

# Инициализация и загрузка индексов при импорте модуля
bm25_indexer = BM25Indexer()
bm25_indexer.load_index(BM25_PICKLE)

word2vec_indexer = Word2VecIndexer()
word2vec_indexer.load_index(W2V_EMBEDDINGS, W2V_MODEL_PATH, CORPUS_PATH)

fasttext_indexer = FastTextIndexer()
fasttext_indexer.load_index(FT_EMBEDDINGS, FT_MODEL_PATH, CORPUS_PATH)

bert_indexer = BERTIndexer(BERT_MODEL_PATH)
bert_indexer.load_index(BERT_EMBEDDINGS, CORPUS_PATH)

# Словарь, сопоставляющий имена индексаторов с их объектами
indexers = {
    'bm25': bm25_indexer,
    'word2vec': word2vec_indexer,
    'fasttext': fasttext_indexer,
    'bert': bert_indexer
}

@app.route('/')
def index():
    return render_template('index.html')  # здесь будет ваше основное описание и кнопка для перехода на страницу поиска

@app.route('/search')
def search():
    return render_template('search.html')  # здесь будет форма для ввода поискового запроса и возможно выбор индексатора

@app.route('/results', methods=['POST'])
def results():
    search_query = request.form['query']
    selected_indexer_key = request.form['indexer']  # Это строка, такая как "bm25", "word2vec" и т.д.

    # Получаем соответствующий объект индексатора из словаря
    selected_indexer = indexers[selected_indexer_key]

    start_time = time.time()
    # Теперь вы можете использовать выбранный индексатор для поиска
    results = selected_indexer.search(search_query)
    end_time = time.time()

    search_time = end_time - start_time

    return render_template('results.html', query=search_query, results=results, time=search_time)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
