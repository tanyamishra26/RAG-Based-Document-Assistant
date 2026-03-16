from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


def format_docs(docs):
    if not docs:
        return "No relevant documents found."

    context = "\n\n".join(doc.page_content for doc in docs)
    return context[:4000]   # prevent token overflow


def create_rag_chain(retriever):

    prompt = ChatPromptTemplate.from_template(
"""
Answer the question using ONLY the context.

If the answer is not in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}
"""
)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "docs": retriever
            }
        )
        | RunnableLambda(
            lambda x: {
                "answer": (prompt | llm | StrOutputParser()).invoke(
                    {
                        "context": x["context"],
                        "question": x["question"]
                    }
                ),
                "sources": x["docs"]
            }
        )
    )

    return rag_chain