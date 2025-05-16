# Biocerto.AI - RAG (Render Ready)

Versione FastAPI di Biocerto.AI con Retrieval-Augmented Generation.

## Funzionalità

- Legge documenti PDF da `/data`
- Genera embeddings con HuggingFace + FAISS
- Risponde a domande con un LLM open-source (`flan-t5-small`)
- Espone API `/ask`
- Include Swagger UI su `/docs`

## Come usarlo

1. Carica i tuoi PDF in `data/`
2. Deploya su [https://render.com](https://render.com)
3. Chiama l'API POST `/ask` con:

```json
{
  "query": "Quali certificazioni biologiche sono incluse?"
}
```
