from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st


def format_docs(docs):
    if not docs:
        return "No relevant documents found."
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever):

    prompt = ChatPromptTemplate.from_template(
"""
You are a document assistant.

Answer the question ONLY using the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}
"""
)

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        huggingfacehub_api_token=st.secrets["HF_TOKEN"],
        temperature=0.2,
        max_new_tokens=512
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
                    {"context": x["context"], "question": x["question"]}
                ),
                "sources": x["docs"]
            }
        )
    )

    return rag_chain