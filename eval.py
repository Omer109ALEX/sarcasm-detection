import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
import my_process_data as mpd
import my_rag as mrag
from langchain_groq import ChatGroq
#from langchain_huggingface import HuggingFaceEmbeddings
import math
import time
import itertools


# Load variables from .env file
load_dotenv()

# Fetch the API key from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
sarcasm_key = os.getenv("GROQ_API_KEY_SARCASM")
gilat_key = os.getenv("GROQ_API_KEY_GILAT")
my_key = os.getenv("GROQ_API_KEY_MY")
rotem_key = os.getenv("GROQ_API_KEY_ROTEM")
openu_key = os.getenv("GROQ_API_KEY_OPENU")
walla_key = os.getenv("GROQ_API_KEY_WALLA")


# Initialize embeddings
#embedding_name = "sentence-transformers/all-roberta-large-v1"
#embedding = HuggingFaceEmbeddings(model_name=embedding_name)

numbers = [0,1,2,3,4,5,6]
#models = ["llama-3.1-70b-versatile", "llama3-70b-8192", "llama-3.1-8b-instant"]
#models = ["llama3-70b-8192", "llama-3.1-8b-instant"]
models = ["llama3-70b-8192"]
#models = ["llama-3.1-70b-versatile"]
#models = ["llama-3.1-8b-instant"]

#keys = [walla_key, openu_key, my_key, sarcasm_key, rotem_key, gilat_key]
keys =[]
for n in numbers:
    name = f'G_KEY{n}'
    keys.append(os.getenv(name))
minute_wait_time = 60
hour_wait_time = 60 * minute_wait_time
day_wait_time = 12 * hour_wait_time

def create_llm(model_name, key):
    llm = ChatGroq(model=model_name, temperature=0.6, groq_api_key=key, model_kwargs={
        "top_p": 0.7,
        "seed": 109,
        "response_format": {"type": "json_object"},
    })
    return llm


async def eval_zero_shot(file_path, label_name, key):
    model_index = 0

    while True:
        model_name = models[model_index]
        llm = create_llm(model_name, key)
        print(f'Starting with model {model_name} for {file_path}\n\n ++++++++++++++++++++++++++++++++++\n\n')
        try:
            await asyncio.to_thread(mrag.ask_llm_from_csv_zero_shot, file_path, llm, label_name, wanted_speed=5)
            break  # Exit the function when the file is processed
        except Exception as e:
            model_index = (model_index + 1) % len(models)
            print(f"An unexpected error occurred: {e}")


async def eval_rag(file_path, label_name, key, embedding, embedding_name):
    model_index = 0

    while True:
        model_name = models[model_index]
        llm = create_llm(model_name, key)
        print(f'Starting with model {model_name} for {file_path}\n\n ++++++++++++++++++++++++++++++++++\n\n')
        try:
            await asyncio.to_thread(mrag.ask_llm_from_csv_rag, file_path, embedding, embedding_name, llm, label_name)
            break  # Exit the function when the file is processed
        except Exception as e:
            model_index = (model_index + 1) % len(models)
            print(f"An unexpected error occurred: {e}")


async def eval_rag_all_models(file_path, label_name, llm, context_coloumn):
    while True:
        try:
            await asyncio.to_thread(mrag.ask_llm_from_csv_rag_with_context, file_path, llm, label_name, context_coloumn, wanted_speed=20)            
            break  # Exit the function when the file is processed
        except Exception as e:
            e_name = type(e).__name__                
            if e_name == "BadRequestError":
                pass   
            elif e_name == "InternalServerError":
                pass                             
            elif e_name == "ServiceUnavailableError":
                pass 
            elif e_name == "TooManyRequestsError":
                pass
            elif str(e) == f'!!!!!!!!!!!!!!!The response was too long go spleep!!!!!!!!!!!!!!!!':
                await asyncio.sleep(day_wait_time)  # Non-blocking sleep 
            elif e_name == "RateLimitError":
                # request \ token day limit
                if ("Limit 1000000" in str(e)) or ("Limit 14,400" in str(e)): 
                    print(f"\nError with {file_path} and {llm.model_name}, go sleep for hour\n ")
                    await asyncio.sleep(hour_wait_time)  # Non-blocking sleep
                # request \ token minute limit                        
                elif ("Limit 131072" in str(e)) or ("Limit 30" in str(e)): 
                    print(f"\nError with {file_path} and {llm.model_name}, go sleep for minute\n ")
                    await asyncio.sleep(minute_wait_time)  # Non-blocking sleep                                                                                      
            else:
                print(f"Exception type: {e_name}") 
                print(f"\nError with {file_path} and {llm.model_name}: {e}\n")


async def to_run_zero_shot(files, labels, keys):
    try:
        tasks = [eval_zero_shot(file_path, label_name, key) for file_path, label_name, key in zip(files, labels, keys)]
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"An error occurred during processing: {e}")


async def to_run_rag(files, labels, keys, embedding, embedding_name):
    try:
        tasks = [eval_rag(file_path, label_name, key, embedding, embedding_name) for file_path, label_name, key in zip(files, labels, keys)]
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"An error occurred during processing: {e}")


async def to_run_rag_all_models(files, label_name, llms, context_coloumn):
    try:
        tasks = [eval_rag_all_models(file_path, label_name, llm, context_coloumn) for file_path, llm in zip(files, llms)]
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    
        
async def combine_every_interval(file_paths, output_file_path, interval=300):
    while True:
        await asyncio.sleep(interval)
        mpd.combine_csv(file_paths, output_file_path)



async def main():
    llms = [create_llm(model, key) for model, key in itertools.product(models, keys)]
    # Define your file paths and labels
    in_or_all = "all"
    original_file = f'./data/all/context/data_all_rag_{in_or_all}.csv'
    #original_file = f'./data/all/context/part_3.csv'
    target_file = f'./data/all/context/processed_data_all_rag_{in_or_all}.csv'
    #num_splits = len(keys)
    num_splits = len(llms)

    split_files = mpd.split_csv(original_file, f'rag_{in_or_all}', num_splits)
    label_name = f"rag_{in_or_all}"
    context_coloumn = f"context_{in_or_all}"
    labels = [label_name] * num_splits

    # Start the combining task
    combine_task = asyncio.create_task(combine_every_interval(split_files, target_file))

    #emedding
    #embedding_name = "sentence-transformers/all-roberta-large-v1" # Description: Based on RoBERTa-large, this model has been fine-tuned for various semantic similarity tasks. 
    #embedding = HuggingFaceEmbeddings(model_name=embedding_name) 
    
    # Run the asyncio tasks
    #await to_run_rag(split_files, labels, keys, embedding, embedding_name)
    await to_run_rag_all_models(split_files, label_name, llms, context_coloumn)

    # Final combination after all processing is done
    mpd.combine_csv(split_files, target_file)
    print("Final combination completed.")
    
    # Cancel the combine task once all processing is done
    combine_task.cancel()
    try:
        await combine_task
    except asyncio.CancelledError:
        pass

# Run the main function
asyncio.run(main())