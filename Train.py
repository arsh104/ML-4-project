from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils import load_and_tokenize_dataset

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
dataset = load_and_tokenize_dataset(tokenizer)

# Prepare data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
output_dir = "/content/drive/MyDrive/your_folder/output_model"
training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir=output_dir + "/logs",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    num_train_epochs=7,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=True)
