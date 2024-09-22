import torch
import logging 
import configparser
from fastapi import FastAPI
from pydantic import BaseModel
from rich.logging import RichHandler
from fastapi.responses import JSONResponse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

config = configparser.ConfigParser()

config.read('config.ini')
model_path = output_dir = config.get('serve', 'model_path')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RichHandler()
logger.addHandler(handler)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f'Using Device: {device}')

logger.info(f'Model Path: {model_path}')
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

app = FastAPI()

class ClaimInfo(BaseModel):
    claim_text: str

def get_veracity(text):
    """
    Analyzes the veracity of the provided text using a pretrained model.

    Args:
        text (str): The text to be analyzed for veracity.

    Returns:
        int: The predicted label index, representing the model's classification 
        of the input text.
    """
    logger.info('Tokenizing Text..')
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logger.info('Inferencing..')
        outputs = model(**inputs)
        logger.info('Softmax..')
        probs = outputs[0].softmax(1)
        logger.info('Argmax..')
        pred_label_idx = probs.argmax()
        logger.info('Returning Value..')
        return pred_label_idx.item()
    

@app.post("/claim/v1/predict")
async def predict_veracity(claim_info: ClaimInfo):
    try:
        claim_text = claim_info.claim_text
        logger.info('Check Veracity..')
        veracity = get_veracity(claim_text)
    
        return JSONResponse(
            status_code=200,
            content=
            {   
                'message':'success',
                'veracity':veracity
            }
        )
    except Exception as e:
        logger.error('Error Occured While Processing', exc_info=True)
        return JSONResponse(
            status_code=500,
            content=
            {
                'message': 'Error Occured While Detecting Veracity',
                'veracity':-1
            }
        )
