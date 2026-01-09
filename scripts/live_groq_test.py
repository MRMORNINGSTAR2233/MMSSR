"""Live Groq end-to-end smoke test (safe, small).

Creates a tiny text document, indexes it, retrieves it, and asks the Groq generator a question.

Note: This performs a live call to Groq and may incur costs. The script expects GROQ_API_KEY and LLM_PROVIDER in environment variables.
"""
import logging

from mmssr.parsers import DocumentElement, ModalityType
from mmssr.retriever import MultiModalRetriever, MultiModalVectorStore
from mmssr.embedders import MultiModalEmbedder
from mmssr.generator import MultiModalGenerator

logging.basicConfig(level=logging.INFO)

def main():
    elem = DocumentElement(
        id='test_doc_1',
        type=ModalityType.TEXT,
        content=(
            'Acme Corp was founded in 2010 by Alice and Bob in San Francisco. '
            'They build fintech APIs for small businesses.'
        ),
        metadata={'topic': 'company_history'},
        source='test_corpus',
        page_number=1,
    )

    vs = MultiModalVectorStore(persist_directory='./data/test_chroma')
    embedder = MultiModalEmbedder()
    retriever = MultiModalRetriever(embedder=embedder, vector_store=vs)

    print('Indexing document...')
    retriever.index_documents([elem])

    query = 'When was Acme Corp founded?'
    print('Retrieving top results for query:', query)
    results = retriever.retrieve(query, n_results=5)
    for r in results:
        print('-', r.id, r.type.value, r.source)

    print('\nInitializing Groq generator...')
    gen = MultiModalGenerator(provider='groq')
    print('Provider:', gen.provider, 'Model:', gen.model)

    print('\nGenerating answer... (this will call Groq)')
    try:
        answer = gen.generate(query, results)
        print('\n--- Generated answer ---\n')
        print(answer['answer'])
        print('\n--- Sources ---\n', answer.get('sources'))
    except Exception as e:
        print('Live generation failed:', type(e).__name__, str(e))

if __name__ == '__main__':
    main()
