from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from bs4 import BeautifulSoup
from Stopwords import stopwords
import string

# Funkcja do czyszczenia tekstu
def clean_txt(path='data/Wiersze.txt'):
    sentences = []
    table = str.maketrans('', '', string.punctuation)
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.lower()
            line = line.translate(table)  # Usuń interpunkcję
            soup = BeautifulSoup(line, 'html.parser')
            line = soup.get_text()
            line = line.replace(".", " . ")
            line = line.replace(",", " , ")
            line = line.replace("-", " - ")
            line = line.replace("/", " / ")
            words = line.split()
            
            filtered_sentence = " ".join(word for word in words if word not in stopwords)
            # filtered_sentence = " ".join(word for word in words)

            sentences.append(filtered_sentence)
    
    return sentences

# Pobranie tokenizera i modelu HerBERT
model_name = "allegro/herbert-klej-cased-v1"
tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Czyszczenie tekstu i uzyskanie listy zdań
sentences = clean_txt('data/Wiersze.txt')

# Przekształcenie listy zdań w dataset
dataset = Dataset.from_dict({'text': sentences})

# Podzielenie datasetu na zbiór treningowy i testowy
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Przygotowanie danych do treningu
def preprocess_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

encoded_train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
encoded_test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Definicja argumentów treningowych
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=False
)

# Stworzenie obiektu Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_test_dataset,
)

# Trenowanie modelu
trainer.train()

# Zapisanie wytrenowanego modelu i tokenizera
output_dir = "polish_love_poems_herbert"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
