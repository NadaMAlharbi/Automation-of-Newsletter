from dotenv import load_dotenv
load_dotenv()

import textwrap
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from deep_translator import GoogleTranslator
from langchain.agents import Tool
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup  # لإزالة علامات HTML
import re  # لتنظيف النص باستخدام تعبيرات منتظمة
import requests  # لاستخدام Extractor API


# Fetch documents using Extractor API and clean them
def get_documents_from_web(url):
    api_key = "25ed2d37fd3cd193d679e28dab9bf23ec0e3bd06"
    extractor_url = f"https://extractorapi.com/api/v1/extractor/?apikey={api_key}&url={url}"

    # Send request to the Extractor API
    response = requests.get(extractor_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the full text content from the response
        data = response.json()
        document_text = data.get("text", "")
    else:
        print(f"Failed to fetch the webpage: {response.status_code}")
        return []

    # Clean the text using BeautifulSoup to remove HTML tags
    soup = BeautifulSoup(document_text, "html.parser")
    cleaned_text = soup.get_text()

    # Remove unwanted special characters using regular expressions
    cleaned_text = re.sub(r'[^\w\s.,!?؛،]', '', cleaned_text)

    # Split the text into smaller chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(cleaned_text)

    # Create a list of LangChain-compatible documents
    documents = [Document(page_content=doc, metadata={"source": url}) for doc in docs]

    return documents


def translate_text(text, target_language="en"):
    translator = GoogleTranslator(source='auto', target=target_language)

    # ترجم النص
    translated_text = translator.translate(text)

    # استبدال كلمة "التكنولوجيا" بـ "التقنية" و"اللغة" بـ "اللغوية"
    translated_text = translated_text.replace("تكنولوجيا", "تقنية")
    translated_text = translated_text.replace("اللغة", "اللغوية")

    return translated_text


def translate_title(title):
    """
    Translate the title according to specific rules:
    - For organizations, governments, and companies, the first word should be a noun.
    - For studies or academic reports, start titles with specific phrases.
    - Translate company names and enclose them in parentheses in Arabic.
    """
    # Replace common English terms with Arabic equivalents
    translated_title = GoogleTranslator(source='auto', target='ar').translate(title)

    # استبدال كلمة "تكنولوجيا" بـ "تقنية" وكلمة "اللغة" بـ "اللغوية"
    translated_title = translated_title.replace("تكنولوجيا", "تقنية")
    translated_title = translated_title.replace("اللغة", "اللغوية")

    # قواعد لتنسيق العنوان وفقًا للمعايير المحددة
    if "company" in title.lower():
        translated_title = translated_title.replace("company", "شركة")

    if "research" in title.lower():
        translated_title = "دراسة تكشف " + translated_title

    # إذا كان العنوان يتحدث عن منظمة، يمكن إضافة "وزارة" أو "شركة" بناءً على السياق
    if "government" in title.lower():
        translated_title = "حكومة " + translated_title

    # تأكد من وجود الأقواس حول أسماء الشركات باللغة العربية
    translated_title = re.sub(r'\((.*?)\)', r'(\1)', translated_title)

    # إزالة الأقواس المربعة
    translated_title = translated_title.replace("[", "").replace("]", "")

    return translated_title


def translate_documents(docs, target_language="en"):
    translated_docs = []
    for doc in docs:
        if isinstance(doc, str):
            doc = Document(page_content=doc, metadata={})

        translated_text = translate_text(doc.page_content, target_language)
        doc.page_content = translated_text
        translated_docs.append(doc)
    return translated_docs


# RAG: Create a FAISS-based vector store from the documents
def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore


# Define general questions for extracting key information from the article
general_questions = [
    "What is the main topic discussed in this article?",
    "Who is involved in the actions or events mentioned?",
    "What are the key findings or advancements?",
    "Why is this topic significant or important?",
    "What challenges or controversies are mentioned?"
]


def extract_key_info(article):
    # Initialize the language model with gpt-4o-mini
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    # Create a vector store from the article
    vectorstore = create_vectorstore(article)

    # Use the vector store as the retriever
    retriever = vectorstore.as_retriever()

    # Set up the retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # Extract answers to general questions
    results = {}
    for question in general_questions:
        results[question] = qa_chain.invoke(question)

    return results


# Define the summarization template with your specific structure
summarize_template = """
Please provide a structured summary of this article, adhering to these guidelines:

1. List all relevant statistics mentioned in the article, ensuring that all numbers appear in parentheses (e.g., 50 -> (50)).

2. Provide a concise summary of the article using the following structure:

   a. Adjust the title to start with a noun or entity relevant to the article, based on these Arabic title guidelines:
      - For organizations, governments, and companies, the first word should be a noun (e.g., "وزارة" (Ministry), "شركة" (Company), etc.).
      - For studies or academic reports, start titles with phrases such as "باحثون يطورون" (Researchers develop), "دراسة تكشف" (Study reveals), أو "تطوير نظام" (Development of a system).
      - Ensure that the title reflects the core idea of the article without sensationalism.

   b. If the article mentions a company name in English, translate it into Arabic and place it in parentheses, e.g., "شركة (جوجل) تطلق خدمة جديدة للذكاء الاصطناعي".

   c. For research articles, reflect the key findings or outcomes of the study in the title.

   d. Ensure that any relevant numbers in the summary are placed in parentheses (e.g., "زاد الإنتاج بنسبة (30%)").

3. Provide a four-sentence summary that follows this structure:
   - **First Sentence**: Explain what was developed, achieved, or discovered.
   - **Second Sentence**: Briefly describe the functionality or purpose.
   - **Third Sentence**: Mention key results or findings, with important numbers in parentheses.
   - **Fourth Sentence**: Indicate any future plans, goals, or developments.

4. Keep the summary under 100 words (excluding the category and title). 

Format your response as follows:

[Title in English]

[Empty Line]

[Four-sentence summary in English]

News Article:
{context}

Summary:
"""

# Create the summarization prompt template
summarization_prompt = ChatPromptTemplate.from_template(summarize_template)


# Tools for Agent
def create_summarization_chain(docs):
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=summarization_prompt  # Pass the prompt as a ChatPromptTemplate
    )
    return chain.invoke({"context": docs})


# Initialize the OpenAI model with gpt-4o-mini
model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.4
)

# Define tools for the agent to use
doc_loader_tool = Tool(
    name="DocumentLoader",
    func=get_documents_from_web,
    description="Fetches documents from the web using Extractor API."
)

summarize_tool = Tool(
    name="Summarizer",
    func=lambda docs: create_summarization_chain(docs),
    description="Summarizes content from the documents."
)

translate_tool = Tool(
    name="Translator",
    func=translate_documents,
    description="Translates documents into Arabic."
)

# Define a prompt for the agent with required variables
agent_prompt = PromptTemplate(
    input_variables=["input", "tools", "agent_scratchpad", "tool_names"],
    template="""
    You are an assistant designed to handle web documents. Given a URL, you will perform the following tasks:

    1. Fetch the document from the web.
    2. Summarize the content.
    3. Translate the summary into Arabic.

    The tools you have access to are: {tool_names}.
    Use the tools in the correct order. Show your work in the scratchpad.

    Input: {input}
    Tools: {tools}
    Agent Scratchpad: {agent_scratchpad}
    """
)

# Use the new agent constructor method and provide the prompt
agent = create_react_agent(
    tools=[doc_loader_tool, summarize_tool, translate_tool],
    llm=model,
    prompt=agent_prompt  # Provide the prompt to the agent
)

# List of URLs to process
urls = [
    'https://techcrunch.com/2024/08/14/mit-researchers-release-a-repository-of-ai-risks/?guccounter=1'
]


# Process and print the title and translated summary for each URL
def process_and_print_translations(urls):
    for i, url in enumerate(urls):
        print(f"\nProcessing URL {i + 1}: {url}\n")

        # Fetch documents from the web using the URL
        docs = get_documents_from_web(url)

        # Create the document chain for summarization
        chain = create_stuff_documents_chain(
            llm=model,
            prompt=summarization_prompt  # Use summarization_prompt instead of string
        )

        # Get the summary of the original content
        response = chain.invoke({
            "context": docs  # Passing the original documents to summarize
        })

        # Get the title and summary
        title_and_summary = response.split("\n", 1)  # Split into title and the rest of the summary
        title = title_and_summary[0].strip()  # First line as the title
        summary = title_and_summary[1].strip() if len(title_and_summary) > 1 else ""

        # Format the English summary for readability
        formatted_summary = textwrap.fill(summary, width=150)

        # Print the title and summary in English
        print("Summary in English:")
        print(f"{title}\n")
        print(f"{formatted_summary}\n")

        # Translate the title and summary into Arabic
        translated_title = translate_title(title)  # Use the new function to translate title
        translated_summary = translate_text(summary, target_language="ar")

        # Ensure the same format for the translated summary
        formatted_translated_summary = textwrap.fill(translated_summary, width=150)

        # Print the title and summary in Arabic with the same format
        print("\nSummary in Arabic:")
        print(f"{translated_title}\n")
        print(f"{formatted_translated_summary}\n")


# Process and print translations for all URLs
process_and_print_translations(urls)
