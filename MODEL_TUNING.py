import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import os

def fine_tune_model(training_file="training_prompts.txt", output_dir="./fine_tuned_model"):
    """Fine-tune GPT-2 on the provided dataset."""
    
    # Disable wandb to avoid API key issues
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Prepare dataset
    print("Preparing dataset...")
    with open(training_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(prompts)} prompts from {training_file}")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128, padding="max_length")
    
    dataset = Dataset.from_dict({'text': prompts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"Train dataset size: {len(split_dataset['train'])}")
    print(f"Test dataset size: {len(split_dataset['test'])}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Check device and set appropriate precision
    device_type = str(model.device).split(':')[0]
    use_fp16 = torch.cuda.is_available() and device_type == 'cuda'
    use_bf16 = device_type == 'xla'  # For TPU
    
    print(f"Device: {model.device}")
    print(f"Using fp16: {use_fp16}")
    print(f"Using bf16: {use_bf16}")
    
    # Training arguments - FIXED VERSION
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",  # CHANGED FROM evaluation_strategy
        eval_steps=500,
        logging_steps=100,
        learning_rate=5e-5,
        fp16=use_fp16,
        bf16=use_bf16,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],  # Disable all reporting
        save_safetensors=True,  # Use safetensors format
        dataloader_pin_memory=False,  # Reduce memory usage
    )
    
    # Trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset['test'],
        data_collator=data_collator
    )
    
    # Fine-tune
    print("Starting fine-tuning...")
    print("This may take 15-30 minutes depending on your hardware...")
    
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Save model
        print(f"Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"✅ Fine-tuned model saved to {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise e

def test_model(model_path="./fine_tuned_model", test_prompt="Create a fantasy story about"):
    """Test the fine-tuned model with a sample prompt."""
    try:
        print(f"Loading fine-tuned model from {model_path}...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        
        print(f"Testing with prompt: '{test_prompt}'")
        inputs = tokenizer.encode(test_prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=100, 
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")

if __name__ == "__main__":
    # Fine-tune the model
    fine_tune_model()
    
    # Test the fine-tuned model
    print("\n" + "="*60)
    print("TESTING THE FINE-TUNED MODEL")
    print("="*60)
    test_model()