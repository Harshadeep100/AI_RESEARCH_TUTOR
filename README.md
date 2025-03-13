# AI_RESEARCH_TUTOR
🚀 An AI-powered tutor that synthesizes knowledge from research papers on CI/CD integration of machine learning models in various cloud services.

📌 Project Overview
This project builds an AI-driven research assistant that extracts, processes, and retrieves knowledge from 60+ research papers related to CI/CD integration for machine learning models in various cloud platforms. It enables users to interactively query these research papers and receive detailed, structured, and contextual responses generated by a TinyLlama-based model.

📖 Key Features
✅ Automated PDF Processing – Extracts text from research papers using PyPDFLoader.
✅ Chunking & Vectorization – Splits extracted text into meaningful chunks and converts them into embeddings using sentence-transformers/all-MiniLM-L6-v2.
✅ Efficient Knowledge Retrieval – Stores the embeddings in a FAISS vector database for fast and accurate semantic search.
✅ Used Mistral 7b GGUF, TinyLlama, phi-2, and GPT-2 models for Answer Generation – Uses GPT-Generated Unified Format to generate structured answers.
✅ Advanced Query Processing – Implements MMR-based retrieval (Maximum Marginal Relevance) to ensure relevant research context is used.
✅ Structured & Contextual Answers – Uses an optimized prompt template to ensure responses are comprehensive, structured, and well-organized.
✅ Interactive Q&A Mode – Allows users to interactively ask questions and receive AI-generated insights from the research papers.

🛠️ Tech Stack
🔹Python
🔹LangChain (PyPDFLoader, RecursiveCharacterTextSplitter, RetrievalQA)
🔹FAISS (Facebook AI Similarity Search)
🔹Sentence Transformers (all-MiniLM-L6-v2)
🔹CTransformers (TinyLlama/Mistral Model)
🔹Transformers (AutoModelForCausalLM, AutoTokenizer)

⚙️ How It Works
🔹Load & Process PDFs: Extracts text from research papers.
🔹Chunking & Embedding: Splits text into manageable chunks and converts them into embeddings.
🔹Vector Store Creation: Saves embeddings in FAISS for fast retrieval.
🔹Question Answering Pipeline: Retrieves the most relevant research context and generates a structured answer using TinyLlama/Mistral.
🔹Interactive Chat Mode: Users can ask questions interactively.

📌 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Harshadeep100/AI_RESEARCH_TUTOR.git
cd AI_Research_Assistant
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Research Assistant
python ai_research_assistant.py

📌 Future Enhancements
🔹 Expand Research Database: Add more research papers related to emerging ML DevOps trends.
🔹 Improve Model Selection: Experiment with more lightweight models for faster inference.
🔹 Deploy as API/Web App: Build a Flask or FastAPI backend and integrate with a React-based UI.
🔹 Multi-Query Support: Allow batch querying for more comprehensive research analysis.

🙌 Contributing
Contributions are welcome! Feel free to submit issues or create pull requests. 🚀
