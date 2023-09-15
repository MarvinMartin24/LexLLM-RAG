import json
import os
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple
from langchain.prompts import PromptTemplate
import requests
from threading import Lock
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
import gradio as gr


class PostDataFetcher:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.response_data = []

    def fetch_data(self) -> List[Dict[str, Any]]:
        max_page = False
        page = 1
        while not max_page:
            url = f"{self.base_url}?page={page}"
            response = requests.get(url)
            if response.status_code == 200:
                page += 1
                page_data = response.json()
                for item in page_data:
                    if '-transcript' in item.get('link', ''):
                        print(page - 1, item['id'], item['link'])
                        post_id = item['id']
                        post_link = item['link']
                        post_url = f"{self.base_url}/{post_id}"
                        post_response = requests.get(post_url)
                        if post_response.status_code == 200:
                            post_content = post_response.json().get('content', '')
                            self.response_data.append({'id': post_id, 'link': post_link, 'content': post_content})
                        else:
                            print(f"Failed to fetch data for post {post_id}. Status code: {post_response.status_code}")
            else:
                max_page = True
                print(f"Failed to fetch data for page {page}. Status code: {response.status_code}")
        return self.response_data

class ConversationParser:
    def __init__(self, response_data: List[Dict[str, Any]]):
        self.response_data = response_data
        self.all_conversations = []

    def parse_conversations(self) -> List[Dict[str, Any]]:
        for i, data in enumerate(self.response_data):
            html_content = data['content']['rendered']
            soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
            podcast_info_text = soup.find(text=lambda text: "Podcast #" in text)
            podcast_number = f"#{podcast_info_text.split('#')[-1].strip()}"
            podcast_url = soup.find("a", text="full YouTube version of the podcast").get("href")
            ts_segment_elements = soup.find_all(class_="ts-segment")
            conversation_data = {
                "conversations": []
            }
            previous_speaker = None
            for ts_segment in ts_segment_elements:
                chapter_heading = ts_segment.find_previous("h2", id=True)
                if chapter_heading:
                    current_chapter = chapter_heading.get("id")
                    current_chapter = current_chapter.replace("chapter", "").replace("_", " ").strip()
                ts_name_element = ts_segment.find(class_="ts-name")
                ts_text_element = ts_segment.find(class_="ts-text")
                ts_timestamp_a_element = ts_segment.find(class_="ts-timestamp")
                ts_timestamp_href = ts_timestamp_a_element.find("a")["href"] if ts_timestamp_a_element else ""
                current_speaker = ts_name_element.get_text()
                if current_speaker == "":
                    current_speaker = previous_speaker
                else:
                    previous_speaker = current_speaker
                current_text = str(ts_text_element.get_text()).encode('utf-8').decode('utf-8')
                segment_data = {
                    "speaker": current_speaker,
                    "chapter": current_chapter,
                    "href": ts_timestamp_href.strip(),
                    "text": current_text.strip()
                }
                conversation_data["conversations"].append(segment_data)
                conversation_data["podcast_number"] = podcast_number
                conversation_data["url"] = podcast_url
            self.all_conversations.append(conversation_data)
        return self.all_conversations

    def save_to_jsonl(self, filename: str):
        with open(filename, "w", encoding='utf-8') as jsonl_file:
            for item in self.all_conversations:
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

def fetch_data():
    base_url = "https://lexfridman.com/wp-json/wp/v2/posts"
    data_fetcher = PostDataFetcher(base_url)
    response_data = data_fetcher.fetch_data()
    parser = ConversationParser(response_data)
    parser.parse_conversations()
    parser.save_to_jsonl("/Users/marvinmartin/Desktop/conversation/conversations.json")

class ChatWrapper:
    def __init__(self, chain: ConversationalRetrievalChain):
        self.chain = chain
        self.lock = Lock()

    def __call__(
        self, query: str
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        output = ""
        try:
            # If chain is None, that is because no API key was provided.
            if self.chain:
                response = self.chain({"question": query})
                output = response["answer"]
                sources = response["source_documents"]
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return output, sources

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = HuggingFacePipeline.from_model_id(
        model_id="tiiuae/falcon-7b-instruct",
        task="text-generation",
        model_kwargs={"max_length": 1000, "do_sample": False, "trust_remote_code": True}
    )

    embed_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        query_instruction="Represent the query for retrieval: ",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectordb = Chroma(persist_directory="Chroma", embedding_function=embed_model)

    template = """
    You are a helpful AI assistant and provide the answer for the question based on the given context.

    Context:{context}

    >>QUESTION<<{question}
    >>ANSWER<<
    """.strip()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history",  input_key='question', output_key='answer', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt":prompt}
    )
    return chain

def create_vector_db():
    embed_model = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        query_instruction="Represent the query for retrieval: ",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    def metadata_func(record: dict, metadata: dict) -> dict:
        metadata["speaker"] = record.get("speaker")
        metadata["chapter"] = record.get("chapter")
        metadata["href"] = record.get("href")
        return metadata

    loader = JSONLoader(
        file_path='/Users/marvinmartin/Desktop/conversation/conversations.json',
        jq_schema='.conversations[]',
        content_key="text",
        metadata_func=metadata_func,
        json_lines=True
    )
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    data = loader.load()
    page_splits = text_splitter.split_documents(data)
    vectordb = Chroma(persist_directory="Chroma", embedding_function=embed_model)
    vectordb.add_documents(documents=page_splits)
    return vectordb

if __name__ == "__main__":
    # # Usage
    # fetch_data()
    # create_vector_db()
    chain = load_chain()
    chat = ChatWrapper(chain)  
    
    block = gr.Blocks(css=".gradio-container {background-color: black}")
    sources_list = []
    with block:
        with gr.Row():
            gr.Markdown("<h3><center>LexLLM: LLM Train on Latest Lex Fridman Podcatst</center></h3>")

        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            bot_message, sources_list = chat(message)
            print(bot_message)
            print(sources_list)
            sources_list_text = "".join(
                [
                    "\nSources:\n"
                ] 
                + 
                [
                    os.linesep + '  -> ' + str(source.metadata['href']) + ' (' + str(source.metadata['speaker']) + ') ' + str(source.page_content[:min(len(source.page_content), 200)]) + " ... " + os.linesep for source in sources_list
                ]
            )
            bot_message = f"{bot_message}{os.linesep}{sources_list_text}"
            chat_history.append((message, bot_message))
            return "", chat_history
        
        def clear_chat():
            global chat
            chat = ChatWrapper(load_chain())  

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat)

    block.launch(debug=True)
    

    # query = "What's George Hotz definition of AGI?"
    # result = chat(query)
    # print(result)
    # query = "According to him, When do you think we will reach it?"
    # result = chat(query)
    # print(result)
    # Who is George Hotz?

   