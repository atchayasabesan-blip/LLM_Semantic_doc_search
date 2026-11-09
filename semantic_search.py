
import chromadb
import csv
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


# 1. Database connection

DB_PATH = r"D:\LLM TASKS\LLM_Vector_Embeddings\chroma_db"

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name="ai_education_documents"
)


# 2. Create folder structure

DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(exist_ok=True)

DOCUMENTS_DIR = DATASET_DIR / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)


# 3. Embedding function

embedding_function = DefaultEmbeddingFunction()


class ChromaEmbeddings(Embeddings):

    def embed_documents(self, texts):
        return embedding_function(texts)

    def embed_query(self, text):
        return embedding_function([text])[0]


# 4. Create 10 new documents

documents = [
    {
        "id": "week12_doc_01",
        "title": "Artificial Intelligence in Education",
        "category": "Artificial Intelligence",
        "content": "Artificial intelligence helps students learn through smart learning platforms, automated assessments, and personalized recommendations. AI tools support teachers by analyzing student performance and providing useful learning suggestions."
    },
    {
        "id": "week12_doc_02",
        "title": "Natural Language Processing",
        "category": "NLP",
        "content": "Natural Language Processing allows computers to understand and process human language. It is used in chatbots, language translation, speech recognition, and text summarization. NLP improves communication between humans and computers."
    },
    {
        "id": "week12_doc_03",
        "title": "Student Performance Analysis",
        "category": "Student Analytics",
        "content": "Student performance analysis uses academic marks, attendance, assignment results, and examination scores. Educational institutions use this information to identify students who need additional support and improve learning outcomes."
    },
    {
        "id": "week12_doc_04",
        "title": "Data Privacy in Education",
        "category": "Data Privacy",
        "content": "Data privacy protects student personal information, academic records, and attendance details. Educational institutions should use secure databases, access controls, encryption, and responsible data management practices."
    },
    {
        "id": "week12_doc_05",
        "title": "Online Learning Platforms",
        "category": "Online Learning",
        "content": "Online learning platforms allow students to attend classes, access study materials, submit assignments, and participate in discussions through the internet. They provide flexible learning opportunities from different locations."
    },
    {
        "id": "week12_doc_06",
        "title": "Personalized Learning Systems",
        "category": "Personalized Learning",
        "content": "Personalized learning systems recommend educational content according to individual student abilities, interests, and learning speed. Artificial intelligence can identify knowledge gaps and suggest suitable lessons."
    },
    {
        "id": "week12_doc_07",
        "title": "AI Chatbots for Students",
        "category": "Chatbots",
        "content": "AI chatbots help students by answering academic questions, explaining difficult topics, providing learning resources, and offering instant assistance. They use natural language processing to understand student questions."
    },
    {
        "id": "week12_doc_08",
        "title": "Bias in Artificial Intelligence",
        "category": "AI Ethics",
        "content": "AI bias occurs when an artificial intelligence system produces unfair results because of biased training data or algorithms. Using diverse datasets, testing models, and monitoring outcomes can reduce bias."
    },
    {
        "id": "week12_doc_09",
        "title": "Cloud Computing in Education",
        "category": "Cloud Computing",
        "content": "Cloud computing provides online storage, software, and computing resources for educational institutions. Students and teachers can access learning materials, share files, and collaborate from different devices."
    },
    {
        "id": "week12_doc_10",
        "title": "Cybersecurity in Online Education",
        "category": "Cybersecurity",
        "content": "Cybersecurity protects online educational platforms, student accounts, and academic information from unauthorized access. Strong passwords, multi-factor authentication, regular updates, and secure networks improve protection."
    }
]


# 5. Create TXT files and prepare new documents

existing_data = collection.get(
    include=["documents", "metadatas"]
)

existing_ids = set(existing_data["ids"])

new_ids = []
new_texts = []
new_metadatas = []

for item in documents:

    file_path = DOCUMENTS_DIR / (
        item["id"].replace("week12_doc_", "document_") + ".txt"
    )

    # Create TXT document
    file_path.write_text(
        item["content"],
        encoding="utf-8"
    )

    # Add only if ID does not already exist
    if item["id"] not in existing_ids:

        new_ids.append(item["id"])
        new_texts.append(item["content"])

        new_metadatas.append({
            "title": item["title"],
            "category": item["category"],
            "source": file_path.name
        })

    print("Document created:", file_path)


# 6. Add new documents to existing ChromaDB

if new_ids:

    new_embeddings = embedding_function(new_texts)

    collection.add(
        ids=new_ids,
        documents=new_texts,
        metadatas=new_metadatas,
        embeddings=new_embeddings
    )

    print("\nNew documents added:", len(new_ids))

else:
    print("\nAll 10 documents already exist in ChromaDB.")


# 7. Connect LangChain with ChromaDB

vector_store = Chroma(
    client=client,
    collection_name="ai_education_documents",
    embedding_function=ChromaEmbeddings()
)


# 8. Fifteen search queries

queries = [
    "How does artificial intelligence help students learn?",
    "How can computers understand human language?",
    "How can teachers identify weak students?",
    "How is student information protected?",
    "How do online classes help students?",
    "How can lessons be personalized?",
    "What are the uses of AI chatbots?",
    "How can AI bias be reduced?",
    "How does cloud computing help education?",
    "How can online learning systems be secured?",
    "How does machine learning improve education?",
    "How can student attendance be analyzed?",
    "How does automatic translation work?",
    "How can technology support teachers?",
    "How can students improve their learning performance?"
]


# 9. Retrieve all documents for keyword search

data = collection.get(
    include=["documents", "metadatas"]
)

all_documents = []

for i, document in enumerate(data["documents"]):

    all_documents.append({
        "document": document,
        "metadata": data["metadatas"][i] or {}
    })


# 10. Result lists

semantic_results = []
similarity_scores = []
keyword_results = []


# 11. Perform semantic and keyword searches

for query_number, query in enumerate(queries, start=1):

    print("\nQuery:", query)

    # Semantic search - Top 5

    results = vector_store.similarity_search_with_score(
        query,
        k=min(5, collection.count())
    )

    for rank, (doc, distance) in enumerate(results, start=1):

        metadata = doc.metadata

        similarity = 1 / (1 + float(distance))

        semantic_results.append([
            query_number,
            query,
            rank,
            metadata.get("title", "N/A"),
            metadata.get("category", "N/A"),
            metadata.get("source", "N/A"),
            round(float(distance), 4),
            round(similarity, 4),
            doc.page_content
        ])

        similarity_scores.append([
            query_number,
            query,
            rank,
            round(float(distance), 4),
            round(similarity, 4)
        ])

        if rank <= 2:
            print(
                "Semantic:",
                metadata.get("title", "N/A"),
                "| Score:",
                round(similarity, 4)
            )

    # Keyword-based search

    query_words = set(
        re.findall(r"[a-zA-Z0-9]+", query.lower())
    )

    matched_documents = []

    for item in all_documents:

        text = (
            item["document"] + " " +
            str(item["metadata"].get("title", "")) + " " +
            str(item["metadata"].get("category", ""))
        ).lower()

        document_words = set(
            re.findall(r"[a-zA-Z0-9]+", text)
        )

        matches = query_words.intersection(document_words)

        score = len(matches) / max(len(query_words), 1)

        if score > 0:
            matched_documents.append((score, item))

    matched_documents.sort(
        key=lambda x: x[0],
        reverse=True
    )

    for rank, (score, item) in enumerate(
        matched_documents[:5],
        start=1
    ):

        metadata = item["metadata"]

        keyword_results.append([
            query_number,
            query,
            rank,
            metadata.get("title", "N/A"),
            metadata.get("category", "N/A"),
            round(score, 4),
            item["document"]
        ])


# 12. Save CSV files inside dataset folder

def save_csv(filename, headers, rows):

    file_path = DATASET_DIR / filename

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)

    print("\nCreated:", file_path)


# Search results

save_csv(
    "search_results.csv",
    [
        "Query_Number", "Query", "Rank", "Title",
        "Category", "Source", "Distance",
        "Similarity_Score", "Document"
    ],
    semantic_results
)


# Similarity scores

save_csv(
    "similarity_scores.csv",
    [
        "Query_Number", "Query", "Rank",
        "Distance", "Similarity_Score"
    ],
    similarity_scores
)


# Keyword results

save_csv(
    "keyword_results.csv",
    [
        "Query_Number", "Query", "Rank",
        "Title", "Category",
        "Keyword_Match_Score", "Document"
    ],
    keyword_results
)


# Queries

save_csv(
    "queries.csv",
    ["Query_Number", "Query"],
    [
        [i, query]
        for i, query in enumerate(queries, start=1)
    ]
)


# 13. Final output

print("\n======================================")
print("ALL FILES CREATED SUCCESSFULLY")
print("======================================")

print("Total Queries:", len(queries))
print("Total Documents:", len(documents))
print("Total ChromaDB Records:", collection.count())
print("Semantic Results:", len(semantic_results))
print("Similarity Scores:", len(similarity_scores))
print("Keyword Results:", len(keyword_results))
