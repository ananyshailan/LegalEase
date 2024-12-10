import requests
from bs4 import BeautifulSoup
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Set up Google API key for Generative AI
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Function to search Indian Kanoon
def search_indian_kanoon(query):
    search_url = f"https://indiankanoon.org/search/?formInput={query.replace(' ', '+')}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.content, "html.parser")

    results = []
    for result in soup.find_all('div', class_='result_title'):
        title = result.find('a').get_text()
        link = "https://indiankanoon.org" + result.find('a')['href']
        results.append({'title': title, 'link': link})
        
    return results

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to set up Langchain QA chain
def get_conversational_chain():
    prompt_template = """
    You are a legal assistant tasked with extracting specific details from a case document. 
    Answer the question as accurately as possible based on the provided context. 
    If the answer is not available, say, "Answer is not available in the context."
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Function to process user input and query the PDFs
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    # Load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Fetch more relevant documents

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

# Function to ask for a brief case summary
def get_case_summary():
    return "What is this case about in 20 words?"

# Streamlit App
def main():
    # Inject custom CSS for positioning the title in the top-left corner
    # st.markdown(
    #     """
    #     <style>
    #     .top-left {
    #         position: absolute;
    #         top: 0px;  /* Adjusted to the top */
    #         left: 0px;  /* Adjusted to the left */
    #         font-size: 24px;
    #         font-weight: bold;
    #         color: #4B4B4B;
    #     }
    #     </style>
    #     <div class="top-left">Legal Ease</div>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.title("Legal Ease")  # Updated title
    st.markdown("### AI Research Assistant: A tool to streamline legal research and enhance productivity.")

    # Step 1: Upload PDFs
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if pdf_files:
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDFs processed successfully! Now ask a question or search for similar cases.")

    # Step 2: Choose to ask from PDFs or search
    option = st.radio(
        "Would you like to ask a question from your PDFs or search?",
        ('Ask PDF', 'Search')
    )

    question = st.text_input("Ask a question:")

    if option == 'Ask PDF':
        if st.button("Get Answer from PDF"):
            if not pdf_files:
                st.error("Please upload a PDF first.")
            elif question:
                with st.spinner("Searching your PDFs..."):
                    answer = user_input(question)
                    st.write(f"**Answer from PDF:** {answer}")
            else:
                st.error("Please ask a question.")
    elif option == 'Search':
        if st.button("Get Search Results"):
            if question:
                with st.spinner("Searching..."):
                    results = search_indian_kanoon(question)
                    if results:
                        st.write(f"**Found {len(results)} similar cases:**")
                        for result in results:
                            st.write(f"[{result['title']}]({result['link']})")
                    else:
                        st.write("No results found.")
            else:
                st.error("Please ask a question.")
    
    # New Feature: Search with the generated case summary
    if st.button("Search Similar Cases"):
        if pdf_files:
            with st.spinner("Generating case summary..."):
                summary_question = get_case_summary()
                summary_answer = user_input(summary_question)
                st.write(f"**Generated Case Summary:** {summary_answer}")
                
                with st.spinner("Searching..."):
                    results = search_indian_kanoon(summary_answer)
                    if results:
                        st.write(f"**Found {len(results)} similar cases:**")
                        for result in results:
                            st.write(f"[{result['title']}]({result['link']})")
                    else:
                        st.write("No results found.")
        else:
            st.error("Please upload a PDF first.")

if __name__ == "__main__":
    main()
