import gradio as gr
import whisper
import tempfile
import os
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone,Weaviate,FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader

os.environ["OPENAI_API_KEY"]="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"

embeddings = OpenAIEmbeddings()

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp:
        temp.write(result["text"])
        temp.flush()
        # Call run_llm function after transcription
        run_llm(temp.name)
        return temp.name

def run_llm(temp_file):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    loader = TextLoader(temp_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    docsearch = FAISS.from_documents(documents, embeddings)
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), docsearch.as_retriever(), return_source_documents=True)

    with gr.Blocks() as demo:
        chatbot=gr.Chatbot()
        msg=gr.Textbox()
        clear=gr.Button("Clear")
        chat_history=[]

        #define function
        def user(user_message, history):
            print("user message:", user_message)
            print("Chat history", history)

            chat_history_str = "\n".join([f"{msg[0]}: {msg[1]}" for msg in chat_history])

            #Get response from QA chain
            response=qa({"question":user_message, "chat_history":chat_history_str})
            #Append user messgae and response to chat history
            history.append((user_message, response["answer"]))
            print ("Updated chat history:", history )
            return gr.update(value=""), history

        msg.submit(user,[msg,chatbot],[msg,chatbot], queue=False)

        clear.click(lambda:None, None, chatbot, queue = False)

        demo.launch(share=True)

def main():
    audio_input = gr.inputs.Audio(source="upload", type="filepath")
    output_text = gr.outputs.Textbox()

    iface = gr.Interface(fn=transcribe_audio, inputs=audio_input,
                         outputs=output_text, title="TRANSCRIPTION APP",
                         description="Upload an audio file and hit the 'Submit' button")

    iface.launch()

if __name__ == '__main__':
    main()
