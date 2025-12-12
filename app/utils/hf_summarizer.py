import os
from app.utils import cfg_logging
from huggingface_hub import InferenceClient

os.environ["HF_TOK"]="<enter token>"
api_key=os.getenv("HF_TOK") 

if api_key is None:
    cfg_logging.logger.error("Error: SYS_VAR 'HF_TOKEN' is not set")
    
def summarize(text:str)->str:  #type:ignore
    client=InferenceClient(
        provider="hf-inference",
        api_key=api_key)
    result=[client.summarization(text,model="Falconsai/text_summarization",)]
    cfg_logging.logger.info("<hf_summarizer> Summarized text.")
    #print("rrrrrrr")
    return result[0]['summary_text']

    
