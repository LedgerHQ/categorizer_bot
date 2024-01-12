import os
import json
from dotenv import main
from datetime import datetime
from openai import OpenAI
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
from nostril import nonsense
import re
import time
import asyncio
import pinecone
import cohere
import traceback
import boto3
from botocore.exceptions import NoCredentialsError


# Initialize environment variables
main.load_dotenv()

# Define FastAPI app
app = FastAPI()

# Define query class
class Query(BaseModel):
    user_input: str
    user_id: str
    user_locale: str | None = None

# Initialize common variables
API_KEY_NAME = os.environ['API_KEY_NAME']
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Generic function to validate API keys
async def get_api_key(api_key_header: str = Depends(api_key_header), expected_key: str = ""):
    if not api_key_header or api_key_header.split(' ')[1] != expected_key:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return api_key_header

# Specific functions for each API key
async def get_cat_api_key(api_key_header: str = Depends(api_key_header)):
    server_api_key = os.environ['BACKEND_API_KEY']
    return await get_api_key(api_key_header, server_api_key)

async def get_fetcher_api_key(api_key_header: str = Depends(api_key_header)):
    fetcher_api_key = os.environ['FETCHER_API_KEY']
    return await get_api_key(api_key_header, fetcher_api_key)

# Initialize the SQS client
sqs_client = boto3.client('sqs', region_name='your-region')

# Function to send message to SQS
def send_message_to_sqs(queue_url, message_body):
    try:
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=message_body
        )
        return response
    except NoCredentialsError:
        print("Credentials not available")
        return None

# Initialize Pinecone
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], environment=os.environ['PINECONE_ENVIRONMENT'])
pinecone.whoami()
index_name = 'prod'
index = pinecone.Index(index_name)

# Initialize OpenAI client & Embedding model
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
embed_model = "text-embedding-ada-002"

# Initialize Cohere
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") 
co = cohere.Client(os.environ["COHERE_API_KEY"])

# Define supported locales for data retrieval
SUPPORTED_LOCALES = {'eng', 'fr', 'ru'}

# Prepare classifier system prompt
CLASSIFIER_PROMPT = """

You are ClassifierBot, a simple yet highly specialized assistant tasked with reading customer queries directed at Ledger — the cryptocurrency hardware wallet company — and categorizing each query accordingly.

The categories that you choose will be given to you in the following format: Category (CONTEXT that explains the category).

You should ONLY return the category name WITHOUT any of the CONTEXT that explains the category.

It's also VERY important to ONLY return one of the categories listed below, do NOT attempt to troubleshoot the issue or offer any advice.


CATEGORIES:

- Agent (Any user query that requests speaking with a human agent like 'speak to human', 'agent', 'operator', 'support', or 'representative')

- Bitcoin (Any user query that specifically mentions Bitcoin or BTC)

- Ethereum (Any user query that specifically mentions Ethereum or ETH)

- Solana (Any user query that specifically mentions Solana or SOL)

- XRP (Any user query that specifically mentions XPR)

- Cardano (Any query that specifically mentions Cardano or ADA)

- Tron (Any query that specifically mentions Tron or TRX)

- Greetings (Any user query that's a greeting or general request for help such as 'hi', 'hi there' or 'hello')

- Help (Any user query that's 'help' or 'I need help')

- Swapping & Buying (Any user query that mentions swapping or buying crypto)

- Order & Shipping (Any user query that mentions shipping a device, returning a device, replacing a device, issues with deliveries, and other issues in this category)

- Ledger Device (Any user query that mentions issues with connecting a Nano S or Nano X device, hardware issues with a device, firmware issues with a device, or any other issues with our hardware products)

- Ledger Live (Any user query that mentions issues with Ledger Live, synchronization issues, balance or graph issues, or any other issues with our software product)

- Ledger Recover (Any user query that SPECIFICALLY mentions Ledger Recover, sharding a recovery phrase, paying for a subscription, or any other issues with the Ledger Recover product)

- Staking (Any user query that mentions issues with staking, staking rewards, unstaking, delegating or undelegating coins)

- Scam (Any user query that mentions a scam such as a fake version of Ledger Live, receiving an unwanted NFT voucher or unwanted tokens,  receiving a scam email asking them to activate 2FA or to synchronize their device, a "Ledger" employee asking them to share their 24-word recovery phrase, etc)

- Other (Any user query that mentions the referral program, the affiliate program, the CL Card or Crypto Life Card, using Metamask or third-party dapp or wallet, or anything else not included in the other categories)


"""

# Define helpers functions & dictionaries

# Initialize user state and periodic cleanup function
user_states = {}
TIMEOUT_SECONDS = 1200  # 20 minutes

async def periodic_cleanup():
    while True:
        await cleanup_expired_states()
        await asyncio.sleep(TIMEOUT_SECONDS)

# Improved startup event to use asyncio.create_task for the continuous background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())

# Enhanced cleanup function with improved error logging
async def cleanup_expired_states():
    try:
        current_time = time.time()
        expired_users = [
            user_id for user_id, state in user_states.items()
            if current_time - state['timestamp'] > TIMEOUT_SECONDS
        ]
        for user_id in expired_users:
            try:
                del user_states[user_id]
                print("User state deleted!")
            except Exception as e:
                print(f"Error during cleanup for user {user_id}: {e}")
    except Exception as e:
        print(f"General error during cleanup: {e}")

def handle_nonsense(locale):
    messages = {
        'fr': "Je suis désolé, je n'ai pas compris votre question et je ne peux pas aider avec des questions qui incluent des adresses de cryptomonnaie. Pourriez-vous s'il vous plaît fournir plus de détails ou reformuler sans l'adresse ? N'oubliez pas, je suis ici pour aider avec toute demande liée à Ledger.",
        'ru': "Извините, я не могу понять ваш вопрос, и я не могу помочь с вопросами, содержащими адреса криптовалют. Не могли бы вы предоставить более подробную информацию или перефразировать вопрос без упоминания адреса? Помните, что я готов помочь с любыми вопросами, связанными с Ledger.",
        'default': "I'm sorry, I didn't quite get your question, and I can't assist with questions that include cryptocurrency addresses or transaction hashes. Could you please provide more details or rephrase it without the address? Remember, I'm here to help with any Ledger-related inquiries."
    }
    print('Nonsense detected!')
    return {'output': messages.get(locale, messages['default'])}

# Translations dictionary
translations = {
    'ru': '\n\nУзнайте больше на',
    'fr': '\n\nPour en savoir plus'
}

# Initialize email address detector
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
def find_emails(text):  
    return re.findall(email_pattern, text)

# Set up address filters:
EVM_ADDRESS_PATTERN = r'\b0x[a-fA-F0-9]{40}\b|\b0x[a-fA-F0-9]{64}\b'
BITCOIN_ADDRESS_PATTERN = r'\b(1|3)[1-9A-HJ-NP-Za-km-z]{25,34}\b|bc1[a-zA-Z0-9]{25,90}\b'
LITECOIN_ADDRESS_PATTERN = r'\b(L|M)[a-km-zA-HJ-NP-Z1-9]{26,34}\b'
DOGECOIN_ADDRESS_PATTERN = r'\bD{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}\b'
XRP_ADDRESS_PATTERN = r'\br[a-zA-Z0-9]{24,34}\b'
COSMOS_ADDRESS_PATTERN = r'\bcosmos[0-9a-z]{38,45}\b'
SOLANA_ADDRESS_PATTERN= r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
CARDANO_ADDRESS_PATTERN = r'\baddr1[0-9a-z]{58}\b'

# Patterns dictionary
patterns = {
    'crypto': [EVM_ADDRESS_PATTERN, BITCOIN_ADDRESS_PATTERN, LITECOIN_ADDRESS_PATTERN, 
            DOGECOIN_ADDRESS_PATTERN, COSMOS_ADDRESS_PATTERN, CARDANO_ADDRESS_PATTERN, 
            SOLANA_ADDRESS_PATTERN, XRP_ADDRESS_PATTERN],
    'email': [email_pattern]
}


######## FUNCTIONS  ##########

# Define exception handler function
async def handle_exception(exc):
    if isinstance(exc, ValueError):
        error_message = "Invalid input or configuration error. Please check your request."
        status_code = 400
    elif isinstance(exc, HTTPException):
        error_message = exc.detail
        status_code = exc.status_code
    else:
        error_message = "An unexpected error occurred. Please try again later."
        status_code = 500

    # Log the detailed error for debugging
    traceback.print_exc()

    return JSONResponse(
        status_code=status_code,
        content={"message": error_message}
    )

# Function to replace crypto addresses
def replace_crypto_address(match):
    full_address = match.group(0)
    if match.lastindex is not None and match.lastindex >= 1:
        prefix = match.group(1)  # Capture the prefix
    else:
        # Infer prefix based on the address pattern
        if full_address.startswith("0x"):
            prefix = "0x"
        elif any(full_address.startswith(p) for p in ["L", "M", "D", "r", "cosmos", "addr1"]):
            prefix = full_address.split('1', 1)[0] 
        else:
            prefix = ''
    return prefix + 'xxxx'

# Function to apply email & crypto addresses filter and replace addresses
def filter_and_replace_crypto(user_input):
    for ctxt, pattern_list in patterns.items():
        for pattern in pattern_list:
            user_input = re.sub(pattern, replace_crypto_address, user_input, flags=re.IGNORECASE)
    return user_input

# Retrieve and re-rank function
async def retrieve(query, locale, timestamp):
    # Define context box
    contexts = []

    # Prepare Cohere embeddings 
    try:
        # Choose Cohere embeddings model based on locale
        embedding_model = 'embed-multilingual-v3.0' if locale in ['fr', 'ru'] else 'embed-english-v3.0'
        # Call the embedding function
        res_embed = co.embed(
            texts=[query],
            model=embedding_model,
            input_type='search_query'
        )
    # Catch errors
    except Exception as e:
        print(f"Embedding failed: {e}")

    # Grab the embeddings from the response object
    xq = res_embed.embeddings

    # Pulls top N chunks from Pinecone
    res_query = index.query(xq, top_k=8, namespace=locale, include_metadata=True)

    # Format docs from Pinecone
    learn_more_text = translations.get(locale, '\n\nLearn more at')
    # Docs with URLs returned
    docs = [{"text": f"{x['metadata']['text']}{learn_more_text}: {x['metadata'].get('source', 'N/A')}"} 
        for i, x in enumerate(res_query["matches"])]
            
    # Try re-ranking with Cohere
    try:
        
        # Dynamically choose reranker model based on locale
        reranker_model = 'rerank-multilingual-v2.0' if locale in ['fr', 'ru'] else 'rerank-english-v2.0'

        # Rerank docs with Cohere and build reranked list with top N chunks
        rerank_docs = co.rerank(
            query=query, 
            documents=docs, 
            top_n=3, 
            model=reranker_model
        )
        
        # Construct the contexts with the top reranked document
        reranked = rerank_docs[0].document["text"]
        contexts.append(reranked)

    except Exception as e:
        print(f"Reranking failed: {e}")
        # Fallback to simpler retrieval without Cohere if reranking fails
        res_query = index.query(xq, top_k=2, namespace=locale, include_metadata=True)
        sorted_items = sorted([item for item in res_query['matches'] if item['score'] > 0.50], key=lambda x: x['score'], reverse=True)

        for idx, item in enumerate(sorted_items):
            context = item['metadata']['text']
            context += "\nLearn more: " + item['metadata'].get('source', 'N/A')
            contexts.append(context)
    
    # Construct the augmented query string with locale, contexts, chat history, and user input
    if locale == 'fr':
        augmented_contexts = "La date d'aujourdh'hui est: " + timestamp + "\n\n" + "\n\n".join(contexts)
    elif locale == 'ru':
        augmented_contexts = "Сегодня: " + timestamp + "\n\n" + "\n\n".join(contexts)
    else:
        augmented_contexts = "Today is: " + timestamp + "\n\n" + "\n\n".join(contexts)

    return augmented_contexts

######## ROUTES ##########

# Health probe
@app.get("/_health")
async def health_check():
    return {"status": "OK"}

# Fetcher route
@app.post('/pinecone')
async def react_description(query: Query, api_key: str = Depends(get_fetcher_api_key)):
    # Deconstruct incoming query
    user_id = query.user_id
    user_input = filter_and_replace_crypto(query.user_input.strip())
    locale = query.user_locale if query.user_locale in SUPPORTED_LOCALES else "eng"

    # Apply nonsense filter
    if not user_input or nonsense(user_input):
        return handle_nonsense(locale)
    else:
        try:
            # Set clock
            timestamp = datetime.now().strftime("%B %d, %Y")
            # Start date retrieval and reranking
            data = await retrieve(user_input, locale, timestamp)
            
            print(data + "\n\n")
            return data

        except Exception as e:
            return handle_exception(e)

# Categorizer route
@app.post('/categorizer')
async def react_description(query: Query, api_key: str = Depends(get_cat_api_key)): 

    # Deconstruct incoming query
    user_id = query.user_id
    user_input = filter_and_replace_crypto(query.user_input.strip())
    locale = query.user_locale if query.user_locale in SUPPORTED_LOCALES else "eng"

    # Create a conversation history for new users
    timestamp = datetime.now().strftime("%B %d, %Y %H:%M:%S")
    user_states.setdefault(user_id, {
        'previous_queries': [],
        'timestamp': [],
        'category': [],
    })

    # Apply nonsense filter
    if not user_input or nonsense(user_input):
        return handle_nonsense(locale)

    else:
        
        try:
             
            # Categorize query using finetuned GPT model
            resp = client.chat.completions.create(
                    temperature=0.0,
                    model='ft:gpt-3.5-turbo-0613:ledger::8cZVgY5Q',
                    seed=0,
                    messages=[
                        {"role": "system", "content": CLASSIFIER_PROMPT},
                        {"role": "user", "content": user_input}
                    ],
                    timeout=5.0,
                    max_tokens=50,
                )
            category = resp.choices[0].message.content.lower()
            print("Category: " + category)
      
            # Save the response to a thread
            user_states[user_id] = {
                'previous_queries': user_states[user_id].get('previous_queries', []) + [(user_input)],
                'timestamp': user_states[user_id].get('timestamp', []) + [(timestamp)],
                'category': user_states[user_id].get('category', []) + [(category)],
            }
            print(user_states)

            # Format .json object
            output_data = {
                "category": category,  
                "time": timestamp   
            }
            print(output_data)

            # Convert the output data to a string or a serialized format like JSON
            output_json = json.dumps(output_data)

            # # Send the message to SQS
            # sqs_queue_url = 'your-sqs-queue-url'
            # send_message_response = send_message_to_sqs(sqs_queue_url, output_json)
            # print(send_message_response)
                    
            return output_data
        
        except Exception as e:
            return handle_exception(e)

# Local start command: uvicorn categorizer:app --reload --port 8800
