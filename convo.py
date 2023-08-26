import os
import git
from queue import SimpleQueue as Queue
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

class DocumentEmbedder:
    def __init__(self, repository_url):
        self.repo_url = repository_url
        self.repo_name = self.repo_url.split('/')[-1].split('.')[0]
        self.local_path = f"hub://user/{self.repo_name}"
        self.chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.query_queue = Queue(maxsize=2)

    def enqueue(self, item):
        if self.query_queue.full():
            self.query_queue.get()
        self.query_queue.put(item)

    def clone_repository(self):
        if not os.path.isdir(self.repo_name):
            git.Repo.clone_from(self.repo_url, self.repo_name)

    def gather_files(self):
        allowed_file_ext = ['.py', '.ipynb', '.md']
        self.documents = []
        for root, _, files in os.walk(self.repo_name):
            for filename in files:
                if os.path.splitext(filename)[1] in allowed_file_ext:
                    try:
                        text_loader = TextLoader(os.path.join(root, filename), encoding='utf-8')
                        self.documents.extend(text_loader.load_and_split())
                    except Exception:
                        continue

    def split_text(self):
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.chunks = splitter.split_documents(self.documents)

    def create_embeddings(self):
        db = Weaviate(dataset_path=self.local_path, embedding_function=self.hf_embeddings)
        db.add_documents(self.chunks)
        self.remove_directory(self.repo_name)
        return db

    def remove_directory(self, dir_path):
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(dir_path)

    def initialize_db(self):
        if weaviate.exists(self.local_path):
            self.db = Weaviate(dataset_path=self.local_path, read_only=True, embedding_function=self.hf_embeddings)
        else:
            self.gather_files()
            self.split_text()
            self.db = self.create_embeddings()

        self.search_engine = self.db.search
        self.search_engine.search_kwargs.update({
            'distance_metric': 'cos',
            'fetch_k': 100,
            'maximal_marginal_relevance': True,
            'k': 3
        })

    def fetch_answers(self, user_query):
        prev_chats = list(self.query_queue.queue)
        qa_chain = ConversationalRetrievalChain.from_llm(self.chat_model, chain_type="stuff", retriever=self.search_engine, condense_question_llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo'))
        response = qa_chain({"question": user_query, "chat_history": prev_chats})
        self.enqueue((user_query, response["answer"]))
        return response['answer']
