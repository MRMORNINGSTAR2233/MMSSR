"""Full RAG end-to-end test (live provider)

- Indexes small text corpus
- Runs retrieval using sentence-transformers embeddings
- Calls Groq generator with retrieved context

Warning: This performs network calls (Hugging Face downloads and Groq API) and may take several minutes and incur API costs.
"""
import logging
import shutil
from pathlib import Path

from mmssr.parsers import DocumentElement, ModalityType
from mmssr.retriever import MultiModalRetriever, MultiModalVectorStore
from mmssr.embedders import MultiModalEmbedder
from mmssr.generator import MultiModalGenerator

logging.basicConfig(level=logging.INFO)

PERSIST_DIR = './data/full_rag_test_chroma'


def clear_persist():
    p = Path(PERSIST_DIR)
    if p.exists():
        print('Removing previous persist directory:', PERSIST_DIR)
        shutil.rmtree(p)


def main():
    clear_persist()

    docs = [
        DocumentElement(
            id='doc_1',
            type=ModalityType.TEXT,
            content="Acme Corp was founded in 2010 by Alice and Bob in San Francisco.",
            metadata={'topic': 'founding'},
            source='doc1',
        ),
        DocumentElement(
            id='doc_2',
            type=ModalityType.TEXT,
            content="In 2015 Acme Corp acquired Beta LLC as part of its expansion.",
            metadata={'topic': 'acquisition'},
            source='doc2',
        ),
        DocumentElement(
            id='doc_3',
            type=ModalityType.TEXT,
            content="Acme Corp focuses on fintech APIs for small businesses and SMBs.",
            metadata={'topic': 'product'},
            source='doc3',
        ),
    ]

    # Initialize components
    vs = MultiModalVectorStore(persist_directory=PERSIST_DIR)
    embedder = MultiModalEmbedder()
    retriever = MultiModalRetriever(embedder=embedder, vector_store=vs)

    print('Indexing documents...')
    retriever.index_documents(docs)

    query = 'When was Acme Corp founded?'
    print('Running retrieval for query:', query)
    results = retriever.retrieve(query, n_results=5)
    for i, r in enumerate(results, 1):
        print(f'Result {i}:', r.id, r.content[:120])

    print('Initializing Groq generator...')
    gen = MultiModalGenerator(provider='groq')
    print('Generator provider/model:', gen.provider, gen.model)

    print('Generating answer using retrieved context...')
    ans = gen.generate(query, results)

    print('\n--- Generated Answer ---\n')
    print(ans['answer'])
    print('\n--- Sources ---\n', ans.get('sources'))

    # Basic check
    if '2010' in ans['answer']:
        print('\nSanity check passed: answer mentions founding year 2010')
    else:
        print('\nSanity check failed: founding year not found in answer')


if __name__ == '__main__':
    main()
