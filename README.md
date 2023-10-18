# Search Engine
Этот проект представляет собой поисковую систему, реализованную в виде веб-приложения. Пользователи могут выполнять поисковые запросы, и система будет предоставлять релевантные результаты из индексированной базы данных с новостными текстами.

## Структура проекта
```bash
- /data                  # папка с данными для индексации и моделями
- /static                # статические файлы веб-приложения
    - style.css          # основные стили проекта
- /templates             # шаблоны HTML для веб-страниц
    - base.html          # базовый шаблон
    - index.html         # главная страница
    - search.html        # страница с формой поиска
    - results.html       # страница с результатами поиска
- app.py                 # основной исполняемый файл веб-приложения
- indexers.py            # файл, содержащий логику индексации
- main.py                # основной исполняемый файл для CLI
- requirements.txt       # зависимости проекта
- README.md              # описание проекта
```

## Веб-приложение
1. Главная Страница:
Страница с приветствием и кнопкой перехода к поиску.
2. Поисковая страница:
Страница с полем для поиска, где пользователи могут ввести свой запрос и выбрать тип индекса.
3. Страница Результатов:
- Показывает время, затраченное на поиск.
- Результаты отображаются в виде списка, каждый элемент которого содержит фрагмент текста, показывающий контекст, в котором было найдено ключевое слово.
- В случае отсутствия результатов выводится соответствующее сообщение.

## CLI Search Tool
CLI Search Tool предоставляет интерфейс командной строки для поиска в корпусе документов с использованием различных методов индексации: BM25, Word2Vec, FastText и BERT.

### Основные элементы
- **BM25Indexer**: Использует алгоритм BM25 для индексации и поиска в корпусе.  
- **Word2VecIndexer**: Использует Word2Vec для векторизации документов и запросов и предварительно обученную модель с сайта RusVectōrēs.
- **FastTextIndexer**: Использует FastText для векторизации документов и запросов и предварительно обученную модель с сайта RusVectōrēs.
- **BERTIndexer**: Использует SBERT для векторизации документов и запросов.

### Использование
1. Установите необходимые зависимости:
```bash
pip install -r requirements.txt
```

2. Для запуска инструмента поиска через командную строку используйте следующую команду:
```bash
python main.py "your query" --indexer=indexer_type
```

### Параметры:
- **your query**: Ваш запрос для поиска.
- **--indexer**: Тип индексатора. Доступные опции: bm25, word2vec, fasttext, bert. По умолчанию используется bm25.

Пример:
```bash
python main.py "искусственный интеллект" --indexer=bm25
```
Этот пример ищет документы, связанные с "искусственным интеллектом", используя индексатор BM25.

**!WARNING:** Чтобы код исправно работал нужно скачать папку с файлами по [ссылке](https://drive.google.com/drive/folders/1w1VkievLj5kdPHrwxY11okAjSxBly78J?usp=drive_link). Эти файлы должны лежать в папке *data*, а Python файлы *main.py* и *indexers.py* — в текущей директории.
