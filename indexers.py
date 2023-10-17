import numpy as np
import time
import pickle
from typing import List, Optional, Union, Tuple, Type
from scipy.sparse import csr_matrix
from rank_bm25 import BM25Okapi
from gensim.models import Word2Vec, FastText, KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pymorphy2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

morph = pymorphy2.MorphAnalyzer()
stops = set(stopwords.words('russian'))

class BaseIndexer:
    def __init__(self, corpus: Optional[List[str]] = None) -> None:
        """
        Инициализация базового индексатора.

        :param corpus: Опциональный список документов для индексации.
        """
        self.corpus = corpus

    def preprocess_text(self, text: str) -> List[str]:
        """
        Предобрабатывает текст, производит лемматизацию и удаляет стоп-слова, возвращая список лемм.

        :param text: Исходный текст.
        :return: Список лемм.
        """
        lemmas = []
        for word in word_tokenize(text):
            if word.isalpha():
                word = morph.parse(word.lower())[0]
                lemma = word.normal_form
                if lemma not in stops:
                    lemmas.append(lemma)
        return lemmas

    def get_query_vector(self, query: str) -> np.ndarray:
        """
        Виртуальный метод для получения вектора запроса. Должен быть переопределен в подклассах.

        :param query: Текстовый запрос.
        :return: Вектор запроса в форме массива numpy.
        :raises NotImplementedError: Если метод не реализован в подклассе.
        """
        raise NotImplementedError("Метод get_query_vector должен быть реализован в подклассе.")

    def get_scores(self, query: str) -> np.ndarray:
        """
        Виртуальный метод для получения оценок релевантности. Должен быть переопределен в подклассах.

        :param query: Текстовый запрос.
        :return: Массив numpy с оценками релевантности.
        :raises NotImplementedError: Если метод не реализован в подклассе.
        """
        raise NotImplementedError("Метод get_scores должен быть реализован в подклассе.")

    def search(self, query: str, top_n: int = 10) -> List[str]:
        """
        Осуществляет поиск наиболее релевантных документов для запроса.

        :param query: Текстовый запрос.
        :param top_n: Количество документов, которое необходимо вернуть.
        :return: Список наиболее релевантных документов.
        """
        start_time = time.time()  # Запоминаем время начала поиска

        cosine_similarities = self.get_scores(query)
        sorted_indices = np.argsort(cosine_similarities)[::-1]
        top_indices = sorted_indices[:top_n]
        top_documents = [self.corpus[i] for i in top_indices]

        end_time = time.time()  # Запоминаем время окончания поиска
        elapsed_time = end_time - start_time  # Вычисляем затраченное время
        
        # Выводим информацию о времени поиска
        print(f"Поиск завершен за {elapsed_time:.2f} секунд.")
        return top_documents

    def save_index(self, filename: str) -> None:
        """
        Виртуальный метод для сохранения индексированных данных в файл. Должен быть переопределен в подклассах.

        :param filename: Имя файла для сохранения данных.
        :raises NotImplementedError: Если метод не реализован в подклассе.
        """
        raise NotImplementedError("Метод save_index должен быть реализован в подклассе.")

    def load_index(self, filename: str) -> None:
        """
        Виртуальный метод для загрузки индексированных данных из файла. Должен быть переопределен в подклассах.

        :param filename: Имя файла для загрузки данных.
        :raises NotImplementedError: Если метод не реализован в подклассе.
        """
        raise NotImplementedError("Метод load_index должен быть реализован в подклассе.")


class BM25Indexer(BaseIndexer):
    def __init__(self, corpus: Optional[List[str]] = None) -> None:
        """
        Инициализация индексатора BM25.

        :param corpus: Опциональный список документов для индексации.
        """
        super().__init__(corpus)
        self.bm25 = None
        self.document_term_matrix: Optional[csr_matrix] = None

    def create_document_term_matrix(self) -> None:
        """Создает Document-Term матрицу с весами BM25."""
        if self.corpus is None:
            raise ValueError("Корпус не был предоставлен.")
        
        tokenized_corpus = [self.preprocess_text(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        # Заполняем матрицу весами BM25
        dense_matrix = np.array([self.bm25.get_scores(tokens) for tokens in tokenized_corpus])
        # Преобразуем матрицу в разреженную форму
        self.document_term_matrix = csr_matrix(dense_matrix)

    def get_document_term_matrix(self) -> csr_matrix:
        """Возвращает Document-Term матрицу с весами BM25."""
        if self.document_term_matrix is None:
            self.create_document_term_matrix()
        return self.document_term_matrix

    def get_query_vector(self, query: str) -> np.ndarray:
        """
        Возвращает вектор запроса с весами BM25.

        :param query: Текстовый запрос.
        :return: Вектор запроса в виде одномерного массива numpy.
        """
        lemmatized_query = self.preprocess_text(query)
        return self.bm25.get_scores(lemmatized_query)

    def get_scores(self, query: str) -> np.ndarray:
        """
        Вычисляет косинусное сходство между запросом и документами в корпусе.

        :param query: Текстовый запрос.
        :return: Одномерный массив numpy, содержащий оценки косинусного сходства.
        """
        query_vector = self.get_query_vector(query)
        # Переводим вектор запроса в разреженный формат для совместимости
        query_vector_sparse = csr_matrix(query_vector)
        document_term_matrix = self.get_document_term_matrix()
        # Вычисляем скалярное произведение
        cosine_similarities = document_term_matrix.dot(query_vector_sparse.T).toarray()
        return cosine_similarities.ravel()  # Сожмем двумерный массив в одномерный

    def save_index(self, filename: str) -> None:
        """
        Сохраняет индексированные данные в файл.

        :param filename: Имя файла для сохранения данных.
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"Индекс успешно сохранен в {filename}")
        except IOError as e:
            print(f"Произошла ошибка ввода-вывода при записи в файл: {e}")
        except Exception as e:
            print(f"Произошла непредвиденная ошибка: {e}")

    def load_index(self, filename: str) -> None:
        """
        Загружает индексированные данные из файла и обновляет атрибуты текущего экземпляра.

        :param filename: Имя файла для загрузки данных.
        """
        try:
            with open(filename, 'rb') as f:
                loaded_obj = pickle.load(f)
            self.__dict__.update(loaded_obj.__dict__)
            print(f"Индекс успешно загружен из {filename}")
        except FileNotFoundError:
            print(f"Файл не найден: {filename}. Пожалуйста, проверьте путь к файлу.")
        except IOError as e:
            print(f"Произошла ошибка ввода-вывода при чтении файла: {e}")
        except pickle.UnpicklingError:
            print("Произошла ошибка при десериализации объекта. Возможно, файл поврежден.")
        except Exception as e:
            print(f"Произошла непредвиденная ошибка: {e}")


class VectorIndexer(BaseIndexer):
    def __init__(self, model: Type[Union[Word2Vec, FastText]], corpus: Optional[List[str]] = None, 
                 pretrained_model: Optional[str] = None, size: int = 100, 
                 window: int = 5, min_count: int = 1, workers: int = 4) -> None:
        """
        Инициализация индексатора векторов.
        
        :param model: Класс модели для использования (Word2Vec или FastText).
        :param corpus: Опциональный список документов для индексации.
        :param pretrained_model: Путь к файлу предварительно обученной модели.
        :param size: Размер вектора в модели.
        :param window: Максимальное расстояние между текущим и прогнозируемым словом в предложении.
        :param min_count: Минимальное количество упоминаний слова в корпусе для его включения в модель.
        :param workers: Количество рабочих потоков для обучения модели.
        """
        super().__init__(corpus)
        self.doc_vectors = None
        self.model_type = model
        self.pretrained_model = pretrained_model
        self.model = None
        self.my_model_path = None

        if corpus:
            self.lemmatized_corpus = [self.preprocess_text(doc) for doc in self.corpus]

            if pretrained_model:
                self.load_pretrained_model()
            else:
                # Если предварительно обученная модель не предоставлена, мы обучаем новую модель
                self.model_params = {
                    'sentences': self.lemmatized_corpus,
                    'vector_size': size,
                    'window': window,
                    'min_count': min_count,
                    'workers': workers
                }
                self.model = self.model_type(**self.model_params)
                self.save_model()

    def load_pretrained_model(self) -> None:
        """
        Загружает предварительно обученную модель в зависимости от типа модели.
        Этот метод должен вызываться только после определения `self.model_type` и `self.pretrained_model`.
        """
        if self.model_type == FastText:
            self.model = KeyedVectors.load(self.pretrained_model)
        elif self.model_type == Word2Vec:
            self.model = KeyedVectors.load_word2vec_format(self.pretrained_model, binary=True)
        else:
            raise ValueError("Неподдерживаемый тип модели")
    
    def save_model(self) -> None:
        """Сохраняет модель после обучения в соответствии с её типом."""
        try:
            if isinstance(self.model, Word2Vec):
                # Сохранение только векторов слов в формате .bin для Word2Vec
                self.model.wv.save_word2vec_format(self.my_model_path, binary=True)
                print(f"Word2Vec модель сохранена в {self.my_model_path}")
            elif isinstance(self.model, FastText):
                # Сохранение полной модели FastText в формате Gensim
                self.model.save(self.my_model_path)
                print(f"FastText модель сохранена в {self.my_model_path}")
            else:
                raise ValueError("Модель должна быть типа Word2Vec или FastText")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")
    
    def get_vector(self, lemmas: List[str]) -> np.ndarray:
        """
        Возвращает усредненный вектор для списка лемм.
        
        :param lemmas: Список лемматизированных слов, для которых требуется получить вектор.
        :return: Усредненный вектор слов из списка.
        """
        if isinstance(self.model, gensim.models.Word2Vec):
            vectors = [self.model.wv[lemma] for lemma in lemmas if lemma in self.model.wv]
        else:  # предполагаем, что это объект KeyedVectors
            vectors = [self.model[lemma] for lemma in lemmas if lemma in self.model]
        
        if not vectors:
            # Если в документе не было ни одного слова из словаря, вернем нулевой вектор
            return np.zeros(self.model.vector_size)
        
        vectors = np.array(vectors)
        document_vector = np.mean(vectors, axis=0)
        return document_vector

    def create_document_vectors_matrix(self) -> None:
        """Создает матрицу векторов документов."""
        if self.corpus is None:
            raise ValueError("Корпус не был предоставлен.")
        self.doc_vectors = np.array([self.get_vector(doc) for doc in self.lemmatized_corpus])

    def get_document_vectors_matrix(self) -> np.ndarray:
        """
        Возвращает матрицу векторов документов.

        :return: Матрица векторов документов.
        """
        if self.doc_vectors is None:
            self.create_document_vectors_matrix()
        return self.doc_vectors

    def get_query_vector(self, query: str) -> np.ndarray:
        """
        Преобразует текстовый запрос в вектор.

        :param query: Текст запроса, который необходимо преобразовать.
        :return: Вектор, представляющий текст запроса.
        """
        lemmatized_query = self.preprocess_text(query)
        return self.get_vector(lemmatized_query)  # Для запроса используем тот же метод, что и для документов

    def get_scores(self, query: str) -> np.ndarray:
        """
        Вычисляет косинусное сходство между запросом и документами.

        :param query: Текст запроса, для которого вычисляются сходства.
        :return: Массив сходств между запросом и каждым документом в индексе.
        """
        query_vector = self.get_query_vector(query).reshape(1, -1)  # reshape для совместимости с cosine_similarity
        doc_vectors = self.get_document_vectors_matrix()
        cosine_similarities = cosine_similarity(query_vector, doc_vectors)
        return cosine_similarities[0]  # возьмем только первый элемент из массива размером [1, кол-во документов]

    def save_index(self, embeddings_filename: str, corpus_filename: str) -> None:
        """
        Сохраняет индексированные данные и корпус в файл.
        
        :param embeddings_filename: Имя файла, куда будут сохранены векторы документов.
        :param corpus_filename: Имя файла, куда будет сохранен корпус.
        """
        try:
            np.save(embeddings_filename, self.doc_vectors)
            with open(corpus_filename, 'w', encoding='utf-8') as f:
                for doc in self.corpus:
                    f.write(doc + '\n')
            print(f"Данные успешно сохранены в {embeddings_filename} и {corpus_filename}")
        except Exception as e:
            print(f"Ошибка при сохранении данных: {e}")

    def load_index(self, embeddings_filename: str, model_filename: str, corpus_filename: str) -> None:
        """
        Загружает индексированные данные, модель и корпус из файла.
        
        :param embeddings_filename: Имя файла, из которого загружаются векторы документов.
        :param model_filename: Имя файла предварительно обученной модели.
        :param corpus_filename: Имя файла, из которого загружается корпус.
        """
        try:
            self.doc_vectors = np.load(embeddings_filename)
            self.pretrained_model = model_filename
            self.load_pretrained_model()
            with open(corpus_filename, 'r', encoding='utf-8') as f:
                self.corpus = [line.strip() for line in f.readlines()]
            print(f"Данные успешно загружены из {embeddings_filename}, {model_filename}, и {corpus_filename}")
        except FileNotFoundError:
            print(f"Файлы не найдены: пожалуйста, проверьте пути {embeddings_filename}, {model_filename}, и {corpus_filename}")
        except Exception as e:
            print(f"Ошибки при загрузке данных: {e}")

class Word2VecIndexer(VectorIndexer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(model=Word2Vec, *args, **kwargs)

class FastTextIndexer(VectorIndexer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(model=FastText, *args, **kwargs)


class BERTIndexer(BaseIndexer):
    def __init__(self, model_name: str, corpus: Optional[List[str]] = None):
        """
        Инициализация индексатора на основе BERT.

        :param model_name: Имя предварительно обученной модели BERT.
        :param corpus: Опциональный список строк для индексации.
        """
        super().__init__(corpus)  # Инициализация базового класса с корпусом

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.corpus_embeddings = None

        if corpus:
            # Индексация исходного корпуса
            self.index(corpus)

    def index(self, corpus: List[str], batch_size: int = 32) -> None:
        """
        Индексация всех документов в корпусе, извлечение их векторных представлений.

        :param corpus: Список строк (документов) для индексации.
        :param batch_size: Размер пакетов для обработки за один раз.
        """
        all_embeddings = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i + batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(**encoded_input).last_hidden_state.mean(dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
        self.corpus_embeddings = np.vstack(all_embeddings)

    def get_scores(self, query: str) -> np.ndarray:
        """
        Вычисление косинусного сходства между запросом и каждым документом в корпусе.

        :param query: Строка запроса.
        :return: Массив сходства для каждого документа в корпусе.
        """
        encoded_input = self.tokenizer([query], padding=True, truncation=True, max_length=128, return_tensors='pt').to(self.device)
        with torch.no_grad():
            query_embedding = self.model(**encoded_input).last_hidden_state.mean(dim=1)
        query_embedding = query_embedding.cpu().numpy()
        scores = cosine_similarity(query_embedding, self.corpus_embeddings).flatten()
        return scores

    def save_index(self, embeddings_filename: str, corpus_filename: str) -> None:
        """
        Сохранение индексированных данных.

        :param embeddings_filename: Имя файла для сохранения векторных представлений.
        :param corpus_filename: Имя файла для сохранения корпуса.
        """
        np.save(embeddings_filename, self.corpus_embeddings)
        with open(corpus_filename, 'w', encoding='utf-8') as f:
            for doc in self.corpus:
                f.write(doc + '\n')

    def load_index(self, embeddings_filename: str, corpus_filename: str) -> None:
        """
        Загрузка индексированных данных.

        :param embeddings_filename: Имя файла с векторными представлениями для загрузки.
        :param corpus_filename: Имя файла с корпусом для загрузки.
        """
        self.corpus_embeddings = np.load(embeddings_filename)
        with open(corpus_filename, 'r', encoding='utf-8') as f:
            self.corpus = [line.strip() for line in f.readlines()]
