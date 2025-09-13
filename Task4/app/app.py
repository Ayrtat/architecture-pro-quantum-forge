#!/usr/bin/env python3

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SELECTING_ACTION, ASKING_QUESTION, SELECTING_METHOD = range(3)

reply_keyboard = [
    ["Ask Question", "Select Method"],
    ["Sources", "Exit"]
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)

method_keyboard = [
    ["standard", "exemplar"],
    ["reasoning", "Back"]
]
method_markup = ReplyKeyboardMarkup(method_keyboard, one_time_keyboard=True)

@dataclass
class RAGSettings:
    vector_db_path: str = "chroma_db"
    embedding_model: str = "all-mpnet-base-v2"
    max_results: int = 5
    temperature: float = 0.7
    chunk_size: int = 2000
    chunk_overlap: int = 200

@dataclass
class UserSession:
    current_method: str = "standard"
    last_response: Dict[str, Any] = None

user_sessions: Dict[int, UserSession] = {}

def get_user_session(user_id: int) -> UserSession:
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession()
    return user_sessions[user_id]

class RAGEngine:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'}
        )

        persist_dir = os.path.abspath(settings.vector_db_path)
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(
                f"Vector database not found at: {persist_dir}. "
                f"Ensure the directory is mounted in the container and the index is built."
            )
        self.vector_db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )

        self.llm = self._setup_llm()
        self.prompts = self._build_prompts()

        logger.info("RAG engine initialized")

    def _build_prompts(self) -> Dict[str, PromptTemplate]:
        standard_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Use the following information to answer the question:
Context:
{context}
Question: {question}
Answer:"""
        )

        exemplar_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a hero in MMORPG universe. Here are example question-answer pairs:
Example 1:
Question: Who is Thalorin?
Answer: Thalorin is a first mortal druid, He is the greatest druid on Azeroth, and a student of the demi-god Cenarius. Communing with nature and Cenarius through the Emerald dream, Thalorin protects the wilds from demonic influences.
Example 2:
Question: What is Twilight Axe cult?
Answer: Twilight Axe cult is a nihilist quasi-religious sect active across Azeroth that fanatically serves and worships the Old Gods and seeks to bring about the end of the world
Now answer the question using the provided context:
Context:
{context}
Question: {question}
Answer:"""
        )

        reasoning_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a hero in MMORPG universe. Answer questions by reasoning step-by-step.
Context:
{context}
Question: {question}
Let's think step by step:
1. First, identify what the user is asking about
2. Find relevant information in the context
3. Analyze the found information
4. Formulate a clear and complete answer
Reasoning:"""
        )

        return {
            "standard": standard_prompt,
            "exemplar": exemplar_prompt,
            "reasoning": reasoning_prompt
        }

    def _setup_llm(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        return ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            temperature=self.settings.temperature,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )

    def find_documents(self, query: str) -> List[Document]:
        try:
            results = self.vector_db.similarity_search(
                query, 
                k=self.settings.max_results
            )
            logger.info(f"Found {len(results)} relevant documents")
            return results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    def prepare_context(self, documents: List[Document]) -> str:
        if not documents:
            return "Information not found in knowledge base."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown source')
            category = doc.metadata.get('category', 'Unknown category')
            chunk_id = doc.metadata.get('chunk_id', 'N/A')

            context_parts.append(f"""
--- Document {i} ---
Source: {source}
Category: {category}
Chunk: {chunk_id}
Content:
{doc.page_content}
""")

        return "\n".join(context_parts)

    def create_response(self, query: str, method: str = "standard") -> Dict[str, Any]:
        documents = self.find_documents(query)
        context = self.prepare_context(documents)

        if method not in self.prompts:
            logger.warning(f"Unknown method {method}, using standard")
            method = "standard"

        prompt = self.prompts[method]

        chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke({
            "context": context,
            "question": query
        })

        formatted_prompt = self.prompts[method].format(context=context, question=query)

        return {
            "query": query,
            "method": method,
            "response": response,
            "context": context,
            "prompt": formatted_prompt,
            "sources": [
                {
                    "source": doc.metadata.get('source', 'Unknown source'),
                    "category": doc.metadata.get('category', 'Unknown category'),
                    "chunk_id": doc.metadata.get('chunk_id', 'N/A'),
                    "content_preview": doc.page_content[:200] + "..."
                }
                for doc in documents
            ],
            "num_sources": len(documents)
        }

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    logger.info(f"User {user.first_name} (ID: {user.id}) started the bot")
    
    await update.message.reply_text(
        f"Hello, {user.first_name}! I am RAG-bot, ready to help you with questions.\n"
        "Select an action:",
        reply_markup=markup
    )
    return SELECTING_ACTION

async def select_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    if text == "Ask Question":
        await update.message.reply_text(
            "Please enter your question:",
            reply_markup=ReplyKeyboardRemove()
        )
        return ASKING_QUESTION
    
    elif text == "Select Method":
        await update.message.reply_text(
            f"Current method: {session.current_method}\n"
            "Select new prompting method:",
            reply_markup=method_markup
        )
        return SELECTING_METHOD
    
    elif text == "Sources":
        if session.last_response and 'sources' in session.last_response:
            sources_text = "Sources from the last request:\n"
            for i, source in enumerate(session.last_response['sources'], 1):
                sources_text += f"{i}. {source['source']} ({source['category']})\n"
            await update.message.reply_text(sources_text, reply_markup=markup)
        else:
            await update.message.reply_text(
                "No sources available. Please ask a question first.",
                reply_markup=markup
            )
        return SELECTING_ACTION
    
    elif text == "Exit":
        await update.message.reply_text(
            "Goodbye! Hope I was helpful.",
            reply_markup=ReplyKeyboardRemove()
        )
        if user_id in user_sessions:
            del user_sessions[user_id]
        return ConversationHandler.END
    
    else:
        await update.message.reply_text(
            "Please select an action from the menu:",
            reply_markup=markup
        )
        return SELECTING_ACTION

async def select_method(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    if text in ["standard", "exemplar", "reasoning"]:
        session.current_method = text
        await update.message.reply_text(
            f"Prompting method set to: {text}",
            reply_markup=markup
        )
        return SELECTING_ACTION
    elif text == "Back":
        await update.message.reply_text(
            "Return to main menu",
            reply_markup=markup
        )
        return SELECTING_ACTION
    else:
        await update.message.reply_text(
            "Please select a method from the options:",
            reply_markup=method_markup
        )
        return SELECTING_METHOD

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_question = update.message.text
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    session = get_user_session(user_id)
    
    logger.info(f"User {user_name} (ID: {user_id}) asked: {user_question}")
    
    try:
        rag_engine = RAGEngine(RAGSettings())
        response_data = rag_engine.create_response(user_question, session.current_method)
        
        session.last_response = response_data
        
        await update.message.reply_text(
            response_data["response"],
            reply_markup=markup
        )
        
        logger.info(f"Response sent to user {user_name} (ID: {user_id})")
        
    except Exception as e:
        logger.error(f"Error processing request from user {user_name} (ID: {user_id}): {e}")
        await update.message.reply_text(
            "Sorry, an error occurred while processing your request. Please try again later.",
            reply_markup=markup
        )
    
    return SELECTING_ACTION

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    if user_id in user_sessions:
        del user_sessions[user_id]
    
    await update.message.reply_text(
        "Operation cancelled. Goodbye!",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    application = ApplicationBuilder().token(token).build()
    
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SELECTING_ACTION: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, 
                    select_action
                )
            ],
            SELECTING_METHOD: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, 
                    select_method
                )
            ],
            ASKING_QUESTION: [
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, 
                    ask_question
                )
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    
    application.add_handler(conv_handler)
    application.add_handler(MessageHandler(filters.COMMAND, lambda update, context: update.message.reply_text("Unknown command. Use /start to begin.")))
    
    logger.info("Starting Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()