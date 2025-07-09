    import torch
    import math
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from utils import load_and_tokenize_dataset
    
    
    model_path = "/content/drive/MyDrive/your_folder/output_model"
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

# Load and tokenize validation dataset
    dataset = load_and_tokenize_dataset(tokenizer, split="validation", max_length=128)
    dataset.set_format(type='torch', columns=['input_ids'])

# Compute perplexity
    losses = []
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch['input_ids'].unsqueeze(0)
            outputs = model(input_ids, labels=input_ids)
            losses.append(outputs.loss.item())
    
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    print(f"Perplexity: {perplexity:.2f}")
