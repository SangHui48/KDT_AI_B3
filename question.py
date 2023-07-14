from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# dosu-beta bot answer
# qa = load_qa_chain(OpenAIChat(model_name="gpt-3.5.-turbo", temperature=0), chain_type="map_rerank", return_intermediate_steps=True)

def get_conversation_chain(model_name, retriever):
    # Original
    model = ChatOpenAI(model_name=model_name)
    qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

    # model = ChatOpenAI(model_name=model_name, temperature=0)
    # qa = ConversationalRetrievalChain.from_llm(model, chain_type="map_rerank", retriever=retriever)
    
    # ChatGPT Answer
    # from langchain.chains import MapReduceDocumentsChain
    # qa = ConversationalRetrievalChain.from_llm(model, combine_docs_chain=MapReduceDocumentsChain(),retriever=retriever)

    return model, qa