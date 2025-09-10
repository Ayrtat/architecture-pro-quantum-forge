# Создание векторного индекса базы знаний

### Эмбеддинг
- Модель: `all-mpnet-base-v2` (репозиторий: [Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2))
- Размер эмбеддингов: 768
- Особенности: 
  - Отличное качество для многоязычных текстов
  - Оптимизирована для семантического поиска
  - Нормализованные эмбеддинги

1. `source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Построение индексов `python3 build_index.py`
4. Тест `python3 search.py`

Примерна статистика 

```json
{
  "model": {
    "name": "all-mpnet-base-v2",
    "embedding_size": 768
  },
  "knowledge_base": {
    "total_documents": 31,
    "total_chunks": 2400+
  },
  "processing": {
    "time_minutes": 4-5,
    "chunks_per_second": 44+
  }
}
```
