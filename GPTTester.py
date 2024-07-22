from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Załaduj wytrenowany model i tokenizer
model_path = "polish_love_poems_herbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Stwórz pipeline do generowania tekstu
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Definiowanie funkcji do generowania wiersza
def generate_poem(prompt, max_length=50, num_return_sequences=1):
    result = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return result[0]['generated_text']

# Generowanie wiersza
prompt = "Miłość jest jak"
poem = generate_poem(prompt, max_length=50, num_return_sequences=1)

# Wyświetlenie wygenerowanego wiersza
print(poem)