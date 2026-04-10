#!/usr/bin/env python3
"""Generate sample documents for local testing when APIs are unavailable."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.models import Document, upsert_documents, load_all_documents, count_documents
from app.text.preprocess import preprocess
from app.indexing.tfidf_index import TfidfIndex
from app.indexing.bm25_index import Bm25Index
from app.indexing.vector_index import VectorIndex

SAMPLE_ARTICLES = [
    {
        "id": "wiki_101",
        "source": "wikipedia",
        "title": "Artificial Intelligence",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "raw_html": "<p>Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills that are associated with the human mind, such as learning and problem-solving. Major AI researchers now reject this definition, equating AI with rational decision-making.</p><p>AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems. As machines become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI. For example, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.</p>",
    },
    {
        "id": "wiki_102",
        "source": "wikipedia",
        "title": "Machine Learning",
        "url": "https://en.wikipedia.org/wiki/Machine_learning",
        "raw_html": "<p>Machine learning (ML) is a field of inquiry devoted to understanding and building methods that learn, that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence. Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, agriculture, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.</p><p>A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers, but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.</p>",
    },
    {
        "id": "wiki_103",
        "source": "wikipedia",
        "title": "Python (programming language)",
        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "raw_html": "<p>Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming. It is often described as a batteries included language due to its comprehensive standard library.</p><p>Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0. Python consistently ranks as one of the most popular programming languages. It is used by many organizations and companies. Major web applications built with Python include Instagram, YouTube, and Reddit.</p>",
    },
    {
        "id": "wiki_104",
        "source": "wikipedia",
        "title": "Web Search Engine",
        "url": "https://en.wikipedia.org/wiki/Web_search_engine",
        "raw_html": "<p>A web search engine or internet search engine is a software system designed to carry out web searches. They search the World Wide Web in a systematic way for particular information specified in a textual web search query. The search results are generally presented in a line of results, often referred to as search engine results pages (SERPs). The information may be a mix of links to web pages, images, videos, infographics, articles, research papers, and other types of files.</p><p>Some search engines also mine data available in databases or open directories. Unlike web directories and social bookmarking sites, which are maintained by human editors, search engines also maintain real-time information by running an algorithm on a web crawler. Web search engines work by storing information about many web pages, which they retrieve from the HTML itself. These pages are retrieved by a web crawler, which is an automated web browser that follows every link on the site.</p>",
    },
    {
        "id": "wiki_105",
        "source": "wikipedia",
        "title": "Information Retrieval",
        "url": "https://en.wikipedia.org/wiki/Information_retrieval",
        "raw_html": "<p>Information retrieval (IR) in computing and information science is the process of obtaining information system resources that are relevant to an information need from a collection of those resources. Searches can be based on full-text or other content-based indexing. Information retrieval is the science of searching for information in a document, searching for documents themselves, and also searching for the metadata that describes data, and for databases of texts, images, or sounds.</p><p>Automated information retrieval systems are used to reduce what has been called information overload. An IR system is a set of algorithms that facilitates the relevance of displayed documents to searched terms. Web search engines are the most visible IR applications. The first automated information retrieval systems were introduced in the 1950s and 1960s.</p>",
    },
    {
        "id": "wiki_106",
        "source": "wikipedia",
        "title": "Natural Language Processing",
        "url": "https://en.wikipedia.org/wiki/Natural_language_processing",
        "raw_html": "<p>Natural language processing (NLP) is an interdisciplinary subfield of computer science and artificial intelligence. It is primarily concerned with providing computers with the ability to process data encoded in natural language and is thus closely related to information retrieval, knowledge representation and computational linguistics. Typically data is collected in text corpora, using either rule-based, statistical, or neural network approaches.</p><p>Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation. Natural language processing has overlap with computational linguistics and can be viewed as a subfield of artificial intelligence. Tasks include text classification, named entity recognition, question answering, sentiment analysis, and machine translation.</p>",
    },
    {
        "id": "wiki_107",
        "source": "wikipedia",
        "title": "Deep Learning",
        "url": "https://en.wikipedia.org/wiki/Deep_learning",
        "raw_html": "<p>Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including natural language processing, speech recognition, computer vision, machine translation, and board and video game programs.</p><p>Artificial neural networks were inspired by information processing and distributed communication nodes in biological systems. ANNs differ from biological brains in several ways including that biological neurons do not all operate by a single mechanism and the computation done by the brain is very different from the back-propagation algorithm used to train most artificial neural networks.</p>",
    },
    {
        "id": "wiki_108",
        "source": "wikipedia",
        "title": "Database",
        "url": "https://en.wikipedia.org/wiki/Database",
        "raw_html": "<p>In computing, a database is an organized collection of data stored and accessed electronically. Small databases can be stored on a file system, while large databases are hosted on computer clusters or cloud storage. The design of databases spans formal techniques and practical considerations, including data modeling, efficient data representation and storage, query languages, security and privacy of sensitive data, and distributed computing issues.</p><p>A database management system (DBMS) is the software that interacts with end users, applications, and the database itself to capture and analyze the data. The DBMS software additionally encompasses the core facilities provided to administer the database. The sum total of the database, the DBMS and the associated applications can be referred to as a database system.</p>",
    },
    {
        "id": "wiki_109",
        "source": "wikipedia",
        "title": "Algorithm",
        "url": "https://en.wikipedia.org/wiki/Algorithm",
        "raw_html": "<p>In mathematics and computer science, an algorithm is a finite sequence of rigorous instructions, typically used to solve a class of specific problems or to perform a computation. Algorithms are used as specifications for performing calculations and data processing. More advanced algorithms can use conditionals to divert the code execution through various routes and deduce valid inferences, achieving automation eventually.</p><p>In contrast, a heuristic is an approach to problem solving that may not be fully specified or may not guarantee correct or optimal results. Algorithms can be expressed in many kinds of notation, including natural languages, pseudocode, flowcharts, drakon-charts, programming languages or control tables.</p>",
    },
    {
        "id": "wiki_110",
        "source": "wikipedia",
        "title": "Cloud Computing",
        "url": "https://en.wikipedia.org/wiki/Cloud_computing",
        "raw_html": "<p>Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user. Large clouds often have functions distributed over multiple locations, each of which is a data center. Cloud computing relies on sharing of resources to achieve coherence and typically uses a pay-as-you-go model, which can help in reducing capital expenses but may also lead to unexpected operating expenses for users.</p><p>The term cloud computing was popularized with Amazon launching its Elastic Compute Cloud product in 2006. Advocates of public and hybrid clouds claim that cloud computing allows companies to avoid or minimize up-front IT infrastructure costs. Cloud computing makes scaling infrastructure easier and faster through autoscaling.</p>",
    },
    {
        "id": "hn_201",
        "source": "hackernews",
        "title": "Show HN: I built a search engine from scratch in Python",
        "url": "https://news.ycombinator.com/item?id=201",
        "raw_html": "<h1>Show HN: I built a search engine from scratch in Python</h1><p>After years of working with Elasticsearch, I decided to build my own search engine from scratch. It uses BM25 for ranking, supports phrase queries, and can index about 10k documents per second on a single machine. The whole thing is under 2000 lines of Python. I learned a ton about inverted indices, skip lists, and query optimization along the way.</p>",
    },
    {
        "id": "hn_202",
        "source": "hackernews",
        "title": "Why TF-IDF still matters in the age of transformers",
        "url": "https://news.ycombinator.com/item?id=202",
        "raw_html": "<h1>Why TF-IDF still matters in the age of transformers</h1><p>Despite the rise of neural ranking models and dense retrieval, TF-IDF remains a crucial baseline. It is interpretable, fast, requires no GPU, and often outperforms more complex methods on keyword-heavy queries. Many production search systems still use TF-IDF or BM25 as a first-stage retriever before neural reranking. Understanding term frequency and inverse document frequency is fundamental to information retrieval.</p>",
    },
    {
        "id": "hn_203",
        "source": "hackernews",
        "title": "Vector databases are overrated for search",
        "url": "https://news.ycombinator.com/item?id=203",
        "raw_html": "<h1>Vector databases are overrated for search</h1><p>Everyone is rushing to build vector search with embeddings, but for most use cases, a well-tuned BM25 implementation gets you 90% of the way there. Vector search shines for semantic similarity and when queries are natural language questions, but for product search, log search, and most enterprise search, lexical methods with good tokenization are hard to beat. The best approach is hybrid: combine BM25 with vector search and let a reranker sort it out.</p>",
    },
    {
        "id": "hn_204",
        "source": "hackernews",
        "title": "Ask HN: What's your stack for full-text search in 2026?",
        "url": "https://news.ycombinator.com/item?id=204",
        "raw_html": "<h1>Ask HN: What's your stack for full-text search in 2026?</h1><p>Looking for recommendations on full-text search solutions. Currently evaluating Elasticsearch, Meilisearch, Typesense, and PostgreSQL full-text search. Our dataset is about 5M documents with lots of structured metadata. Need good relevance tuning, faceted search, and ideally hybrid search with vector capabilities. What are people using in production these days?</p>",
    },
    {
        "id": "hn_205",
        "source": "hackernews",
        "title": "FAISS vs Annoy vs ScaNN: Benchmarking vector search libraries",
        "url": "https://news.ycombinator.com/item?id=205",
        "raw_html": "<h1>FAISS vs Annoy vs ScaNN: Benchmarking vector search libraries</h1><p>I benchmarked the three most popular approximate nearest neighbor search libraries on datasets ranging from 100K to 10M vectors. FAISS with IVF-PQ offers the best balance of speed and recall at scale. Annoy is simplest to use and great for smaller datasets. ScaNN from Google has excellent performance but requires more tuning. For most Python projects under 1M vectors, FAISS IndexFlatIP with normalized embeddings is fast enough and gives exact results.</p>",
    },
    {
        "id": "hn_206",
        "source": "hackernews",
        "title": "The BM25 algorithm explained from first principles",
        "url": "https://news.ycombinator.com/item?id=206",
        "raw_html": "<h1>The BM25 algorithm explained from first principles</h1><p>BM25 (Best Matching 25) is a ranking function used by search engines to estimate the relevance of documents to a given search query. It is based on the probabilistic retrieval framework developed in the 1970s. BM25 builds on TF-IDF by adding document length normalization and term frequency saturation. The key parameters are k1, which controls term frequency saturation, and b, which controls document length normalization. BM25 remains the default ranking function in most search engines including Elasticsearch and Solr.</p>",
    },
    {
        "id": "hn_207",
        "source": "hackernews",
        "title": "Sentence Transformers 5.0: What's new for semantic search",
        "url": "https://news.ycombinator.com/item?id=207",
        "raw_html": "<h1>Sentence Transformers 5.0: What's new for semantic search</h1><p>The latest release of sentence-transformers brings major improvements to semantic search capabilities. New pre-trained models achieve state-of-the-art performance on retrieval benchmarks. The library now supports efficient batch encoding, multi-GPU training, and ONNX export for production deployment. Semantic search works by encoding both documents and queries into dense vector representations, then using cosine similarity or dot product to find the most relevant documents.</p>",
    },
    {
        "id": "hn_208",
        "source": "hackernews",
        "title": "Building a web crawler in Python with asyncio",
        "url": "https://news.ycombinator.com/item?id=208",
        "raw_html": "<h1>Building a web crawler in Python with asyncio</h1><p>Web crawling is the process of systematically browsing the web to collect content and metadata from websites. Modern crawlers use asynchronous I/O to efficiently handle thousands of concurrent connections. Key considerations include respecting robots.txt, implementing politeness delays, handling redirects, deduplicating URLs, and managing the crawl frontier. Python's asyncio with aiohttp makes it straightforward to build a high-performance crawler.</p>",
    },
    {
        "id": "hn_209",
        "source": "hackernews",
        "title": "PageRank is dead, long live learned ranking",
        "url": "https://news.ycombinator.com/item?id=209",
        "raw_html": "<h1>PageRank is dead, long live learned ranking</h1><p>Google's original PageRank algorithm revolutionized web search by using link structure to determine page authority. While the basic PageRank concept is still used, modern search engines rely heavily on machine-learned ranking models that incorporate hundreds of signals including click-through rates, dwell time, content quality, freshness, and user engagement metrics. Neural ranking models like BERT-based cross-encoders can understand query-document relevance at a much deeper level than any single static signal.</p>",
    },
    {
        "id": "hn_210",
        "source": "hackernews",
        "title": "How to evaluate search quality: NDCG, MAP, and MRR explained",
        "url": "https://news.ycombinator.com/item?id=210",
        "raw_html": "<h1>How to evaluate search quality: NDCG, MAP, and MRR explained</h1><p>Evaluating search engine quality requires metrics that capture how well results satisfy user intent. Normalized Discounted Cumulative Gain (NDCG) measures ranking quality by considering both relevance grades and position. Mean Average Precision (MAP) focuses on binary relevance and rewards systems that place relevant documents higher. Mean Reciprocal Rank (MRR) measures how quickly the first relevant result appears. For modern search systems, a combination of offline metrics and online A/B testing with click-through rate provides the most comprehensive evaluation.</p>",
    },
]


def main() -> None:
    print("=== Seeding sample data ===\n")

    docs: list[Document] = []
    for article in SAMPLE_ARTICLES:
        clean_text, tokens = preprocess(article["raw_html"])
        docs.append(
            Document(
                id=article["id"],
                source=article["source"],
                title=article["title"],
                url=article["url"],
                raw_html=article["raw_html"],
                clean_text=clean_text,
                tokens=tokens,
            )
        )

    upsert_documents(docs)
    print(f"Stored {count_documents()} documents in DB.\n")

    all_docs = load_all_documents()
    doc_ids = [d.id for d in all_docs]
    clean_texts = [d.clean_text for d in all_docs]
    token_lists = [d.tokens for d in all_docs]

    print("Building TF-IDF index ...")
    tfidf = TfidfIndex()
    tfidf.build(doc_ids, clean_texts)
    tfidf.save()
    print("  done")

    print("Building BM25 index ...")
    bm25 = Bm25Index()
    bm25.build(doc_ids, token_lists)
    bm25.save()
    print("  done")

    try:
        print("Building vector index ...")
        vec = VectorIndex()
        vec.build(doc_ids, clean_texts)
        vec.save()
        print("  done")
    except Exception as exc:
        print(f"  skipped (model download unavailable): {exc}")

    print(f"\n=== Seeding complete. {len(all_docs)} documents indexed. ===")


if __name__ == "__main__":
    main()
