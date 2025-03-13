import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load and extract text from PDFs
pdf_dir = r"Your_ResearchPapers_Directory"
documents = []
for file in os.listdir(pdf_dir):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_dir, file))
        documents.extend(loader.load())

# Step 2: Split text into small, manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1080, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Step 3: Generate embeddings with sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Store chunks in FAISS vector store
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("research_vector_db")

# Step 5: Load TinyLlama Model (optimized setup)
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_type="mistral",
    context_length=2048,  # ✅ Force model to accept longer inputs
    temperature=0.1,
    max_new_tokens=100000
)

# Step 6: **Improved Prompt for Generalization & Better Answering**
custom_prompt = """
You are a research assistant who synthesizes knowledge from research papers.
Your job is to provide **detailed, structured, and well-organized answers** strictly based on the provided research context.

### **Instructions:**
- Use **all retrieved research papers** to provide a **comprehensive** answer.
- If the retrieved papers provide **partial** information, use them to **generalize the best possible answer**.
- If the answer is **not directly found**, state what **can be inferred** from the research context.
- If the answer is **completely unavailable**, say:  
  "**I could not find this information in the provided research papers, but here’s what we can infer:**"

---

### **Research Context:**
{context}

### **User Question:**
{question}

### **Structured Answer:**
"""

prompt = PromptTemplate(
    template=custom_prompt,
    input_variables=["context", "question"]
)

# Step 7: **Improved Retrieval Setup** (More relevant information)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# Step 8: Function to Ask Queries Interactively (ChatGPT-Like)
def chat(query):
    result = qa_chain.invoke({"query": query})

    print("\n🔍 **Retrieved Context from Database:**\n")
    retrieved_texts = []

    for i, doc in enumerate(result["source_documents"]):
        retrieved_texts.append(doc.page_content)
        print(f"📄 **Source {i + 1}:** {doc.page_content[:500000]}...\n")  # ✅ Show first 500 characters

    # ✅ Ensure context length is within limits by handling it dynamically
    limited_context = "\n\n".join(retrieved_texts[:1])  # ✅ Use only the **most relevant 3 documents**

    # ✅ Final Optimized Prompt for Context-Aware Answering
    structured_prompt = f"""
    You are an AI research assistant that summarizes, synthesizes, and generalizes knowledge from research papers.

    **User Question:**  
    {query}

    **Retrieved Research Context:**  
    {limited_context}

    **Final Answer (Summarized & Generalized):**  
    """

    # ✅ Generate final response
    output = llm.invoke(structured_prompt)
    print("\n🤖 **AI Generated Answer:**\n", output)


# Step 9: Run Interactive Chat Loop
if __name__ == "__main__":
    print("💬 Chat with your Research Papers (type 'exit' to quit)")
    while True:
        query = input("\nYour Question: ")
        if query.lower() == "exit":
            break
        chat(query)
