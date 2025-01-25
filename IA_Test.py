# -*- coding: utf-8 -*-

import re
import warnings
import torch
import threading
import logging
import json
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset, concatenate_datasets, Dataset
from datetime import datetime
import os
import shutil
import pandas as pd
import traceback

# Configure logging
logging.basicConfig(
    filename='chatbot_debug.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
# Used to suppress warnings
warnings.filterwarnings("ignore")

# Paths for model saving, conversation state, and datasets
PERCORSO_MODELLO = '//192.168.1.253/Vol 2/Test/modello_salvato/'   # model data file path
PERCORSO_STATO = './stato_salvato/'       # current state data file path
PERCORSO_DATASET = './dataset/'           # dataset path

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Check if CUDA is available
print("CUDA:", torch.cuda.is_available())

# Number of available GPUs
print("GPU's available:", torch.cuda.device_count())

# GPU name (if available)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

def sanitize_text(text):
    """
    Remove invalid characters from the text to avoid encoding errors,
    while preserving valid accented characters in supported languages.
    """
    sanitized = re.sub(r'[^\w\s.,!?\'’"-]', ' ', text, flags=re.UNICODE)
    return sanitized.strip()

def salva_conversazione(sorgente, messaggio, file_log="conversazione_log.txt"):
    """
    Save conversation logs to file.
    """
    try:
        with open(file_log, "a", encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {sorgente}: {messaggio}\n")
    except Exception as e:
        logging.error(f"Error saving conversation: {e}")

def salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO):
    """
    Save the trained model (knowledge).
    """
    try:
        # Remove the folder if it exists, to avoid conflicts
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        # Save the model in a standard format (disabling safetensors)
        model.save_pretrained(path, safe_serialization=False)
        tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")
        logging.info(f"Model and tokenizer saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model and tokenizer: {e}")

def salva_stato(history, path=PERCORSO_STATO):
    """
    Save the conversation state (history).
    """
    try:
        os.makedirs(path, exist_ok=True)
        # Save conversation history
        with open(os.path.join(path, 'history.pt'), 'wb') as f:
            torch.save(history, f)
        print(f"Conversation history saved in {path}")
        logging.info(f"Conversation history saved in {path}")
    except Exception as e:
        logging.error(f"Error saving conversation state: {e}")

def carica_stato(path=PERCORSO_STATO):
    """
    Load conversation history (state).
    """
    history = []
    try:
        if os.path.exists(os.path.join(path, 'history.pt')):
            with open(os.path.join(path, 'history.pt'), 'rb') as f:
                history = torch.load(f)
            print(f"Conversation history loaded from {path}")
            logging.info(f"Conversation history loaded from {path}")
    except Exception as e:
        logging.error(f"Error loading conversation state: {e}")
    return history

def rimuovi_duplicati(dataset):
    """
    Remove duplicate entries from the dataset.
    """
    try:
        unique_dataset = []
        visto = set()
        for esempio in dataset:
            chiave = (esempio['instruction'], esempio['input'], esempio['output'])
            if chiave not in visto:
                unique_dataset.append(esempio)
                visto.add(chiave)
        logging.info(f"Duplicates removed: {len(dataset) - len(unique_dataset)}")
        return unique_dataset
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return dataset

def valida_dataset(file_path):
    """
    Validate the dataset by checking required keys.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        validi = 0
        non_validi = 0
        for esempio in data:
            if all(key in esempio for key in ['instruction', 'input', 'output']):
                validi += 1
            else:
                non_validi += 1
                logging.warning(f"Invalid example: {esempio}")
        print(f"Valid dataset: {validi} examples")
        print(f"Invalid dataset: {non_validi} examples")
        logging.info(f"Valid dataset: {validi} examples, invalid: {non_validi} examples")
        return validi, non_validi
    except Exception as e:
        logging.error(f"Dataset validation error: {e}")
        return 0, 0

def rimuovi_duplicati_dataset(file_path, output_path):
    """
    Remove duplicates from the dataset and save the cleaned version.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        unique_data = rimuovi_duplicati(data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=4)
        print(f"Dataset without duplicates saved to {output_path}")
        logging.info(f"Dataset without duplicates saved to {output_path}")
    except Exception as e:
        logging.error(f"Error removing duplicates from dataset: {e}")

def carica_modello(modello_salvato=None):
    """
    Load the model and tokenizer from a specified path or use the pre-trained DialoGPT-large model.
    """
    try:
        if modello_salvato and os.path.exists(modello_salvato):
            print(f"Loading model from {modello_salvato}")
            model = AutoModelForCausalLM.from_pretrained(modello_salvato)
            tokenizer = AutoTokenizer.from_pretrained(modello_salvato)
            logging.info(f"Model loaded from {modello_salvato}")
        else:
            print("Loading pre-trained DialoGPT-large model")
            model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
            logging.info("Pre-trained DialoGPT-large model loaded")
    
        # Add a padding token if none is present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logging.info("Padding token added to tokenizer")
    
        tokenizer.clean_up_tokenization_spaces = True
        model.to(device)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise e

def genera_risposta(input_text, model, tokenizer, history=None, max_length=100, temperature=0.7):
    """
    Generate a chatbot response given user input, model, tokenizer, and conversation history.
    """
    try:
        if history is None:
            history = []
    
        # Tokenize user input
        new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
        history.append(new_user_input_ids)
        # Keep only the last 5 exchanges
        history = history[-5:]
    
        # Prepare input for the model
        bot_input_ids = torch.cat(history, dim=-1).to(device)
        attention_mask = torch.ones_like(bot_input_ids).to(device)
    
        # Generate the response
        output_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=bot_input_ids.shape[-1] + max_length,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            no_repeat_ngram_size=2,
            do_sample=True,
            length_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
        # Extract the newly generated part
        response_ids = output_ids[:, bot_input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        history.append(response_ids)
    
        # Sanitize the response
        sanitized_response = sanitize_text(response)
        logging.info(f"Generated response: '{response}'")
        logging.info(f"Sanitized response: '{sanitized_response}'")
        print(f"Generated response (raw): '{response}'")  # Debug
        print(f"Sanitized response: '{sanitized_response}'")  # Debug
    
        # If the response is empty after sanitization, return an error-like message
        if not sanitized_response.strip():
            print("The generated response is empty after sanitization.")
            logging.warning("Empty response after sanitization.")
            return response, history
    
        return response, history

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        print(f"Error generating response: {e}")
        return "Sorry, an error occurred while generating the response.", history

def verifica_tokenizzazione(modello_salvato=PERCORSO_MODELLO, path_dataset=PERCORSO_DATASET, num_esempi=50):
    """
    Test tokenization on a sample from the datasets.
    """
    try:
        model, tokenizer = carica_modello(modello_salvato=modello_salvato)
        
        dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
        if not dataset_paths:
            print("No dataset found for tokenization check.")
            return
        
        sample_texts = []
        for path in dataset_paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for esempio in data[:num_esempi]:
                    instruction = esempio.get('instruction', '').strip()
                    input_text = esempio.get('input', '').strip()
                    output_text = esempio.get('output', '').strip()
                    
                    if input_text:
                        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                    else:
                        prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                    
                    sanitized_prompt = sanitize_text(prompt)
                    sample_texts.append(sanitized_prompt)
        
        tokenized = tokenizer(
            sample_texts,
            truncation=True,
            padding='max_length',
            max_length=100,
            return_tensors='pt'
        )
        
        for i in range(len(sample_texts)):
            print(f"--- Sample {i+1} ---")
            print("Original text:")
            print(sample_texts[i])
            print("\nToken IDs:")
            print(tokenized['input_ids'][i])
            print("\nTokenizer decode:")
            decoded_text = tokenizer.decode(tokenized['input_ids'][i], skip_special_tokens=True)
            print(decoded_text)
            print("\n----------------------\n")
    
    except Exception as e:
        logging.error(f"Error during tokenization check: {e}")
        print(f"Error during tokenization check: {e}")

def analizza_lunghezza_dataset(path_dataset=PERCORSO_DATASET):
    """
    Analyze average and maximum token lengths in the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained('//192.168.1.253/Vol 2/Test/modello_salvato/')
    token_lengths = []
    dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
    for path in dataset_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for esempio in data:
                instruction = esempio.get('instruction', '').strip()
                input_text = esempio.get('input', '').strip()
                output_text = esempio.get('output', '').strip()
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                else:
                    prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                tokenized = tokenizer.encode(prompt, truncation=True, max_length=None)
                token_lengths.append(len(tokenized))
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    print(f"Average sequence length: {avg_length}")
    if token_lengths:
        print(f"Maximum sequence length: {max(token_lengths)}")
    return token_lengths

# REMOVED all text-to-speech functions

# REMOVED all Telegram integration

def addestra_modello(model, tokenizer, path_dataset=PERCORSO_DATASET):
    """
    Train (or continue training) the model on JSON datasets.
    """
    # Check if there's an existing model to continue training
    if os.path.exists('//192.168.1.253/Vol 2/Test/temp_model'):
        print("Loading existing model from //192.168.1.253/Vol 2/Test/temp_model to continue training.")
        model = AutoModelForCausalLM.from_pretrained('//192.168.1.253/Vol 2/Test/temp_model')
        tokenizer = AutoTokenizer.from_pretrained('//192.168.1.253/Vol 2/Test/temp_model')
        model.to(device)
        logging.info("Existing model loaded for continuous training.")
    else:
        print("No existing model found. Training from scratch.")
        logging.info("No existing model found. Training from scratch.")

    # Get the list of JSON files for training
    dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
    if not dataset_paths:
        print("No dataset found for training.")
        logging.warning("No dataset found for training.")
        return model, tokenizer

    datasets = []
    for path in dataset_paths:
        print(f"Loading dataset: {path}")
        try:
            dataset = load_dataset('json', data_files=path, split='train')
            
            def preprocess_examples(examples):
                texts = []
                for instruction, input_text, output_text in zip(examples['instruction'], examples['input'], examples['output']):
                    if input_text.strip():
                        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                    else:
                        prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                    texts.append(prompt)
                return {'text': texts}
            
            dataset = dataset.map(preprocess_examples, batched=True, remove_columns=['instruction', 'input', 'output'])
            datasets.append(dataset)
            print(f"Dataset {path} loaded successfully.")
            logging.info(f"Dataset {path} loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset {path}: {e}")
            logging.error(f"Error loading dataset {path}: {e}")
            continue

    if not datasets:
        print("No dataset was successfully loaded for training.")
        logging.warning("No dataset was successfully loaded for training.")
        return model, tokenizer

    # Concatenate all datasets
    try:
        combined_dataset = concatenate_datasets(datasets)
        logging.info("All loaded datasets concatenated.")
    except Exception as e:
        logging.error(f"Error concatenating datasets: {e}")
        return model, tokenizer

    # Convert to pandas to remove duplicates
    try:
        df = combined_dataset.to_pandas()
        logging.info("Converted dataset to pandas DataFrame for duplicate removal.")
    except Exception as e:
        logging.error(f"Error converting dataset to DataFrame: {e}")
        return model, tokenizer

    # Remove duplicates
    try:
        df_unique = df.drop_duplicates(subset=['text'])
        logging.info("Duplicates removed from concatenated dataset.")
    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
        return model, tokenizer

    # Convert back to a Dataset
    try:
        combined_dataset = Dataset.from_pandas(df_unique)
        logging.info("De-duplicated DataFrame converted back to a Dataset.")
    except Exception as e:
        logging.error(f"Error converting de-duplicated DataFrame back to Dataset: {e}")
        return model, tokenizer

    # Split dataset into train and validation
    try:
        train_val = combined_dataset.train_test_split(test_size=0.1)
        train_dataset = train_val['train']
        eval_dataset = train_val['test']
        logging.info("Dataset split into train and validation.")
    except Exception as e:
        logging.error(f"Error splitting dataset: {e}")
        return model, tokenizer

    # Tokenization
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=100
        )
        # Set labels to be the same as input_ids
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    try:
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        tokenized_val = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        logging.info("Dataset successfully tokenized.")
    except Exception as e:
        logging.error(f"Error tokenizing dataset: {e}")
        return model, tokenizer

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir='//192.168.1.253/Vol 2/Test/temp_model_new',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=2,
        save_strategy='steps',
        save_steps=500,
        evaluation_strategy='steps',
        eval_steps=500,
        learning_rate=5e-6,
        weight_decay=0.01,
        report_to="tensorboard",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Start training
    try:
        print("Starting training...")
        logging.info("Starting training.")
        trainer.train()
        print("Training completed.")
        logging.info("Training completed.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        print(f"Error during training: {e}")
        return model, tokenizer

    # Save the updated model
    try:
        trainer.save_model('//192.168.1.253/Vol 2/Test/temp_model_new')
        tokenizer.save_pretrained('//192.168.1.253/Vol 2/Test/temp_model_new')
        print(f"Updated model saved to //192.168.1.253/Vol 2/Test/temp_model_new")
        logging.info("Updated model and tokenizer saved to //192.168.1.253/Vol 2/Test/temp_model_new.")
    except Exception as e:
        logging.error(f"Error saving updated model: {e}")

    # Rename folders (replace old temp_model with the new one)
    try:
        if os.path.exists('//192.168.1.253/Vol 2/Test/temp_model'):
            shutil.rmtree('//192.168.1.253/Vol 2/Test/temp_model')
        os.rename('//192.168.1.253/Vol 2/Test/temp_model_new', '//192.168.1.253/Vol 2/Test/temp_model')
        print("Updated model saved to //192.168.1.253/Vol 2/Test/temp_model")
        logging.info("Updated model saved to //192.168.1.253/Vol 2/Test/temp_model.")
    except Exception as e:
        print(f"Error replacing temp_model folder: {e}")
        logging.error(f"Error replacing temp_model folder: {e}")

    return trainer.model, tokenizer

def prepara_dataset(file_path, output_path):
    """
    Prepare the dataset by validating, removing duplicates, and saving.
    """
    valida_dataset(file_path)
    rimuovi_duplicati_dataset(file_path, output_path)

# MAIN
if __name__ == "__main__":
    # Example: preparing the dataset (commented out as an example)
    # input_dataset_path = 'human_evolution_dataset.json'
    # output_dataset_path = 'human_evolution_dataset_unici.json'
    # prepara_dataset(input_dataset_path, output_dataset_path)

    # Load the model (pre-trained or previously saved)
    model, tokenizer = carica_modello(modello_salvato=PERCORSO_MODELLO)

    # Load conversation state, if any
    history = carica_stato(path=PERCORSO_STATO)

    print("Welcome back! Type 'esci' to quit.")
    print("Available commands: 'salva stato', 'salva coscienza', 'addestra', 'verifica', 'analizza'")

    while True:
        user_input = input("You: ")
        salva_conversazione("User", user_input)

        if user_input.lower() == "esci":
            print("Chatbot: Arrivederci!")
            salva_conversazione("Chatbot", "Arrivederci!")
            break

        elif user_input.lower() == "salva coscienza":
            salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO)

        elif user_input.lower() == "salva stato":
            salva_stato(history, path=PERCORSO_STATO)

        elif user_input.lower() == "verifica":
            verifica_tokenizzazione()

        elif user_input.lower() == "analizza":
            analizza_lunghezza_dataset()

        elif user_input.lower() == "addestra":
            conferma = input("Are you sure you want to start training? (Y/N): ")
            if conferma.lower() == "y":
                model, tokenizer = addestra_modello(model, tokenizer, path_dataset=PERCORSO_DATASET)
                salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO)
            else:
                print("Training canceled.")

        else:
            risposta, history = genera_risposta(user_input, model, tokenizer, history)
            print("Chatbot:", risposta)
            salva_conversazione("Chatbot", risposta)












'''#Ok ITA

# -*- coding: utf-8 -*-

import re
import warnings
import torch
import threading
import logging
import json
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset, concatenate_datasets, Dataset
from datetime import datetime
import os
import shutil
import pandas as pd
import traceback

# Configure the logging
logging.basicConfig(
    filename='chatbot_debug.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
# Used to not show warnings
warnings.filterwarnings("ignore")

# Path to save the models, the current state e the dataset
PERCORSO_MODELLO = '//192.168.1.253/Vol 2/Test/modello_salvato/'   # model data file path 
PERCORSO_STATO = './stato_salvato/'       # current state data file path
PERCORSO_DATASET = './dataset/'           # dataset path

# Imposta il device (CPU o GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Verifica se CUDA è disponibile
print("CUDA:", torch.cuda.is_available())

# Numero di GPU disponibili
print("GPU's available:", torch.cuda.device_count())

# Nome della GPU (se disponibile)
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

# Funzione per "sanitizzare" il testo, mantenendo i caratteri utili
def sanitize_text(text):
    """
    Rimuove caratteri non validi dal testo per evitare errori di encoding,
    mantenendo però i caratteri accentati validi nelle lingue supportate.
    """
    sanitized = re.sub(r'[^\w\s.,!?\'’"-]', ' ', text, flags=re.UNICODE)
    return sanitized.strip()

# Funzione per salvare i log della conversazione su file
def salva_conversazione(sorgente, messaggio, file_log="conversazione_log.txt"):
    try:
        with open(file_log, "a", encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {sorgente}: {messaggio}\n")
    except Exception as e:
        logging.error(f"Errore nel salvataggio della conversazione: {e}")

# Funzione per salvare il modello (coscienza allenata)
def salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO):
    try:
        # Elimina la cartella se esiste, per evitare conflitti
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        # Salva il modello in formato standard (disabilitando safetensors)
        model.save_pretrained(path, safe_serialization=False)
        tokenizer.save_pretrained(path)
        print(f"Modello e tokenizer salvati in {path}")
        logging.info(f"Modello e tokenizer salvati in {path}")
    except Exception as e:
        logging.error(f"Errore nel salvataggio di modello e tokenizer: {e}")

# Funzione per salvare lo stato della conversazione
def salva_stato(history, path=PERCORSO_STATO):
    try:
        os.makedirs(path, exist_ok=True)
        # Salva la history della conversazione
        with open(os.path.join(path, 'history.pt'), 'wb') as f:
            torch.save(history, f)
        print(f"Storia della conversazione salvata in {path}")
        logging.info(f"Storia della conversazione salvata in {path}")
    except Exception as e:
        logging.error(f"Errore nel salvataggio dello stato della conversazione: {e}")

# Funzione per caricare lo stato della conversazione
def carica_stato(path=PERCORSO_STATO):
    history = []
    try:
        if os.path.exists(os.path.join(path, 'history.pt')):
            with open(os.path.join(path, 'history.pt'), 'rb') as f:
                history = torch.load(f)
            print(f"Storia della conversazione caricata da {path}")
            logging.info(f"Storia della conversazione caricata da {path}")
    except Exception as e:
        logging.error(f"Errore nel caricamento dello stato della conversazione: {e}")
    return history

# Funzione per rimuovere duplicati dal dataset
def rimuovi_duplicati(dataset):
    try:
        unique_dataset = []
        visto = set()
        for esempio in dataset:
            chiave = (esempio['instruction'], esempio['input'], esempio['output'])
            if chiave not in visto:
                unique_dataset.append(esempio)
                visto.add(chiave)
        logging.info(f"Duplicati rimossi: {len(dataset) - len(unique_dataset)}")
        return unique_dataset
    except Exception as e:
        logging.error(f"Errore nella rimozione duplicati: {e}")
        return dataset

# Funzione per validare il dataset
def valida_dataset(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        validi = 0
        non_validi = 0
        for esempio in data:
            if all(key in esempio for key in ['instruction', 'input', 'output']):
                validi += 1
            else:
                non_validi += 1
                logging.warning(f"Esempio non valido: {esempio}")
        print(f"Dataset valido: {validi} esempi")
        print(f"Dataset non valido: {non_validi} esempi")
        logging.info(f"Dataset valido: {validi} esempi, non valido: {non_validi} esempi")
        return validi, non_validi
    except Exception as e:
        logging.error(f"Errore di validazione del dataset: {e}")
        return 0, 0

# Funzione per rimuovere duplicati dal dataset e salvarlo
def rimuovi_duplicati_dataset(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        unique_data = rimuovi_duplicati(data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=4)
        print(f"Dataset senza duplicati salvato in {output_path}")
        logging.info(f"Dataset senza duplicati salvato in {output_path}")
    except Exception as e:
        logging.error(f"Errore nella rimozione duplicati del dataset: {e}")

# Funzione per caricare il modello e il tokenizer
def carica_modello(modello_salvato=None):
    try:
        if modello_salvato and os.path.exists(modello_salvato):
            print(f"Caricamento modello da {modello_salvato}")
            model = AutoModelForCausalLM.from_pretrained(modello_salvato)
            tokenizer = AutoTokenizer.from_pretrained(modello_salvato)
            logging.info(f"Modello caricato da {modello_salvato}")
        else:
            print("Caricamento modello pre-addestrato DialoGPT-large")
            model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-large')
            tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
            logging.info("Modello DialoGPT-large pre-addestrato caricato")
    
        # Aggiunge un token di padding se non presente
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            logging.info("Token di padding aggiunto al tokenizer")
    
        tokenizer.clean_up_tokenization_spaces = True
    
        model.to(device)
        return model, tokenizer
    except Exception as e:
        logging.error(f"Errore nel caricamento del modello: {e}")
        raise e

# Funzione per generare risposte dal chatbot con gestione errori
def genera_risposta(input_text, model, tokenizer, history=None, max_length=100, temperature=0.7):
    try:
        if history is None:
            history = []
    
        # Tokenizza input utente
        new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
        history.append(new_user_input_ids)
        # Manteniamo solo gli ultimi 5 scambi
        history = history[-5:]
    
        # Prepara input per il modello
        bot_input_ids = torch.cat(history, dim=-1).to(device)
        attention_mask = torch.ones_like(bot_input_ids).to(device)
    
        # Genera la risposta
        output_ids = model.generate(
            bot_input_ids,
            attention_mask=attention_mask,
            max_length=bot_input_ids.shape[-1] + max_length,
            temperature=temperature,
            top_p=0.92,
            top_k=50,
            no_repeat_ngram_size=2,
            do_sample=True,
            length_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
        # Estrae la parte generata
        response_ids = output_ids[:, bot_input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        history.append(response_ids)
    
        # Sanitizza la risposta
        sanitized_response = sanitize_text(response)
        logging.info(f"Risposta generata: '{response}'")
        logging.info(f"Risposta sanitizzata: '{sanitized_response}'")
        print(f"Risposta generata (raw): '{response}'")  # Debug
        print(f"Risposta sanitizzata: '{sanitized_response}'")  # Debug
    
        # Se la risposta è vuota dopo la sanitizzazione, ritorna un messaggio di errore
        if not sanitized_response.strip():
            print("La risposta generata è vuota dopo la sanitizzazione.")
            logging.warning("Risposta vuota dopo la sanitizzazione.")
            return response, history
    
        return response, history

    except Exception as e:
        logging.error(f"Errore nella generazione della risposta: {e}")
        print(f"Errore nella generazione della risposta: {e}")
        return "Mi spiace, si è verificato un errore nella generazione della risposta.", history

def verifica_tokenizzazione(modello_salvato=PERCORSO_MODELLO, path_dataset=PERCORSO_DATASET, num_esempi=50):
    try:
        model, tokenizer = carica_modello(modello_salvato=modello_salvato)
        
        dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
        if not dataset_paths:
            print("Nessun dataset trovato per la verifica.")
            return
        
        sample_texts = []
        for path in dataset_paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for esempio in data[:num_esempi]:
                    instruction = esempio.get('instruction', '').strip()
                    input_text = esempio.get('input', '').strip()
                    output_text = esempio.get('output', '').strip()
                    
                    if input_text:
                        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                    else:
                        prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                    
                    # Sanitizza
                    sanitized_prompt = sanitize_text(prompt)
                    sample_texts.append(sanitized_prompt)
        
        tokenized = tokenizer(
            sample_texts,
            truncation=True,
            padding='max_length',
            max_length=100,
            return_tensors='pt'
        )
        
        for i in range(len(sample_texts)):
            print(f"--- Esempio {i+1} ---")
            print("Testo Originale:")
            print(sample_texts[i])
            print("\nToken ID:")
            print(tokenized['input_ids'][i])
            print("\nTokenizzatore Decode:")
            decoded_text = tokenizer.decode(tokenized['input_ids'][i], skip_special_tokens=True)
            print(decoded_text)
            print("\n----------------------\n")
    
    except Exception as e:
        logging.error(f"Errore durante la verifica della tokenizzazione: {e}")
        print(f"Errore durante la verifica della tokenizzazione: {e}")

def analizza_lunghezza_dataset(path_dataset=PERCORSO_DATASET):
    tokenizer = AutoTokenizer.from_pretrained('//192.168.1.253/Vol 2/Test/modello_salvato/')
    token_lengths = []
    dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
    for path in dataset_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for esempio in data:
                instruction = esempio.get('instruction', '').strip()
                input_text = esempio.get('input', '').strip()
                output_text = esempio.get('output', '').strip()
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                else:
                    prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                tokenized = tokenizer.encode(prompt, truncation=True, max_length=None)
                token_lengths.append(len(tokenized))
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    print(f"Lunghezza media delle sequenze: {avg_length}")
    if token_lengths:
        print(f"Lunghezza massima delle sequenze: {max(token_lengths)}")
    return token_lengths

# RIMOSSE tutte le funzioni di text-to-speech

# RIMOSSE tutte le funzioni di integrazione con Telegram

# Funzione di addestramento del modello su dataset in formato JSON
def addestra_modello(model, tokenizer, path_dataset=PERCORSO_DATASET):
    # Carica eventuale modello esistente per continuare l'addestramento
    if os.path.exists('//192.168.1.253/Vol 2/Test/temp_model'):
        print("Caricamento modello esistente da //192.168.1.253/Vol 2/Test/temp_model per continuare l'addestramento.")
        model = AutoModelForCausalLM.from_pretrained('//192.168.1.253/Vol 2/Test/temp_model')
        tokenizer = AutoTokenizer.from_pretrained('//192.168.1.253/Vol 2/Test/temp_model')
        model.to(device)
        logging.info("Modello esistente caricato per addestramento continuo.")
    else:
        print("Nessun modello esistente trovato. Addestramento da zero.")
        logging.info("Nessun modello esistente trovato. Addestramento da zero.")

    # Lista dei path dei file JSON da usare come dataset
    dataset_paths = [os.path.join(path_dataset, f) for f in os.listdir(path_dataset) if f.endswith('.json')]
    if not dataset_paths:
        print("Nessun dataset trovato per l'addestramento.")
        logging.warning("Nessun dataset trovato per l'addestramento.")
        return model, tokenizer

    datasets = []
    for path in dataset_paths:
        print(f"Caricamento dataset: {path}")
        try:
            dataset = load_dataset('json', data_files=path, split='train')
            
            # Prepara dataset con i campi 'text'
            def preprocess_examples(examples):
                texts = []
                for instruction, input_text, output_text in zip(examples['instruction'], examples['input'], examples['output']):
                    if input_text.strip():
                        prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output_text}"
                    else:
                        prompt = f"Instruction: {instruction}\nResponse: {output_text}"
                    texts.append(prompt)
                return {'text': texts}
            
            dataset = dataset.map(preprocess_examples, batched=True, remove_columns=['instruction', 'input', 'output'])
            datasets.append(dataset)
            print(f"Dataset {path} caricato correttamente.")
            logging.info(f"Dataset {path} caricato correttamente.")
        except Exception as e:
            print(f"Errore nel caricamento del dataset {path}: {e}")
            logging.error(f"Errore nel caricamento del dataset {path}: {e}")
            continue

    if not datasets:
        print("Nessun dataset è stato caricato correttamente per l'addestramento.")
        logging.warning("Nessun dataset è stato caricato correttamente per l'addestramento.")
        return model, tokenizer

    # Concatena tutti i dataset
    try:
        combined_dataset = concatenate_datasets(datasets)
        logging.info("Tutti i dataset caricati e concatenati.")
    except Exception as e:
        logging.error(f"Errore nella concatenazione dei dataset: {e}")
        return model, tokenizer

    # Converti in pandas per rimuovere duplicati
    try:
        df = combined_dataset.to_pandas()
        logging.info("Dataset convertito in DataFrame pandas per rimozione duplicati.")
    except Exception as e:
        logging.error(f"Errore nella conversione del Dataset in DataFrame: {e}")
        return model, tokenizer

    # Rimuovi duplicati
    try:
        df_unique = df.drop_duplicates(subset=['text'])
        logging.info("Duplicati rimossi dal dataset concatenato.")
    except Exception as e:
        logging.error(f"Errore nella rimozione dei duplicati: {e}")
        return model, tokenizer

    # Converti di nuovo in Dataset
    try:
        combined_dataset = Dataset.from_pandas(df_unique)
        logging.info("Dataset de-duplicato riconvertito in Dataset.")
    except Exception as e:
        logging.error(f"Errore nella riconversione del DataFrame de-duplicato in Dataset: {e}")
        return model, tokenizer

    # Splitta dataset in train e validation
    try:
        train_val = combined_dataset.train_test_split(test_size=0.1)
        train_dataset = train_val['train']
        eval_dataset = train_val['test']
        logging.info("Dataset suddiviso in train e validation.")
    except Exception as e:
        logging.error(f"Errore nello splitting del dataset: {e}")
        return model, tokenizer

    # Tokenizzazione
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=100
        )
        # Imposta le labels uguali agli input_ids
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    try:
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        tokenized_val = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        logging.info("Dataset tokenizzato correttamente.")
    except Exception as e:
        logging.error(f"Errore nella tokenizzazione del dataset: {e}")
        return model, tokenizer

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir='//192.168.1.253/Vol 2/Test/temp_model_new',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=2,
        save_strategy='steps',
        save_steps=500,
        evaluation_strategy='steps',
        eval_steps=500,
        learning_rate=5e-6,
        weight_decay=0.01,
        report_to="tensorboard",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Avvia l'addestramento
    try:
        print("Inizio dell'addestramento...")
        logging.info("Inizio dell'addestramento.")
        trainer.train()
        print("Addestramento completato.")
        logging.info("Addestramento completato.")
    except Exception as e:
        logging.error(f"Errore durante l'addestramento: {e}")
        print(f"Errore durante l'addestramento: {e}")
        return model, tokenizer

    # Salva il modello aggiornato
    try:
        trainer.save_model('//192.168.1.253/Vol 2/Test/temp_model_new')
        tokenizer.save_pretrained('//192.168.1.253/Vol 2/Test/temp_model_new')
        print(f"Modello aggiornato e salvato in //192.168.1.253/Vol 2/Test/temp_model_new")
        logging.info("Modello e tokenizer aggiornati e salvati in //192.168.1.253/Vol 2/Test/temp_model_new.")
    except Exception as e:
        logging.error(f"Errore nel salvataggio del modello aggiornato: {e}")

    # Rinomina le cartelle (sostituisce la vecchia cartella temp_model con la nuova)
    try:
        if os.path.exists('//192.168.1.253/Vol 2/Test/temp_model'):
            shutil.rmtree('//192.168.1.253/Vol 2/Test/temp_model')
        os.rename('//192.168.1.253/Vol 2/Test/temp_model_new', '//192.168.1.253/Vol 2/Test/temp_model')
        print("Modello aggiornato e salvato in //192.168.1.253/Vol 2/Test/temp_model")
        logging.info("Modello aggiornato e salvato in //192.168.1.253/Vol 2/Test/temp_model.")
    except Exception as e:
        print(f"Errore nella sostituzione della cartella temp_model: {e}")
        logging.error(f"Errore nella sostituzione della cartella temp_model: {e}")

    return trainer.model, tokenizer

# Funzione per preparare il dataset: validazione, rimozione duplicati e salvataggio
def prepara_dataset(file_path, output_path):
    valida_dataset(file_path)
    rimuovi_duplicati_dataset(file_path, output_path)

# MAIN
if __name__ == "__main__":
    # Esempio: preparazione del dataset (commentato come esempio)
    # input_dataset_path = 'human_evolution_dataset.json'
    # output_dataset_path = 'human_evolution_dataset_unici.json'
    # prepara_dataset(input_dataset_path, output_dataset_path)

    # Carica il modello (pre-addestrato o salvato in precedenza)
    model, tokenizer = carica_modello(modello_salvato=PERCORSO_MODELLO)

    # Carica lo stato della conversazione, se esiste
    history = carica_stato(path=PERCORSO_STATO)

    print("Bentornato! Scrivi 'esci' per terminare.")
    print("Comandi disponibili: 'salva stato', 'salva coscienza', 'addestra', 'verifica', 'analizza'")

    while True:
        user_input = input("You: ")
        salva_conversazione("User", user_input)

        if user_input.lower() == "esci":
            print("Chatbot: Arrivederci!")
            salva_conversazione("Chatbot", "Arrivederci!")
            break

        elif user_input.lower() == "salva coscienza":
            salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO)

        elif user_input.lower() == "salva stato":
            salva_stato(history, path=PERCORSO_STATO)

        elif user_input.lower() == "verifica":
            verifica_tokenizzazione()

        elif user_input.lower() == "analizza":
            analizza_lunghezza_dataset()

        elif user_input.lower() == "addestra":
            conferma = input("Sei sicuro di voler avviare l'addestramento? (Y/N): ")
            if conferma.lower() == "y":
                model, tokenizer = addestra_modello(model, tokenizer, path_dataset=PERCORSO_DATASET)
                salva_conoscenza(model, tokenizer, path=PERCORSO_MODELLO)
            else:
                print("Addestramento annullato.")

        else:
            risposta, history = genera_risposta(user_input, model, tokenizer, history)
            print("Chatbot:", risposta)
            salva_conversazione("Chatbot", risposta)'''
