import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random

class PromptGenerator:
    def __init__(self, model_path="./fine_tuned_model"):
        """Initialize the prompt generator."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
    
    def generate_prompt(self, genre="", theme="", tone=""):
        """Generate a writing prompt based on inputs."""
        start_text = "Write a"
        if genre:
            start_text += f" {genre}"
        start_text += " story"
        if theme:
            start_text += f" about {theme}"
        if tone:
            start_text += f" with a {tone} tone"
        start_text += " about"
        
        input_ids = self.tokenizer.encode(start_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 50,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        prompt = self.tokenizer.decode(output[0], skip_special_tokens=True).replace(start_text, "").strip()
        if not prompt.endswith(('.', '!', '?')):
            prompt += '.'
        return prompt.capitalize()
    
    def interactive_session(self):
        """Run an interactive session to generate prompts."""
        print("=== Creative Writing Prompt Generator ===")
        while True:
            print("\nOptions: 1. Custom prompt 2. Random prompt 3. Exit")
            choice = input("Choose (1-3): ").strip()
            if choice == "1":
                genre = input("Genre (e.g., fantasy, or blank): ").strip()
                theme = input("Theme (e.g., love, or blank): ").strip()
                tone = input("Tone (e.g., mysterious, or blank): ").strip()
                prompt = self.generate_prompt(genre, theme, tone)
                print(f"\nPrompt: {prompt}\n")
            elif choice == "2":
                genres = ['fantasy', 'sci-fi', 'mystery', 'horror', 'romance']
                themes = ['love', 'betrayal', 'redemption', 'identity', 'survival']
                tones = ['mysterious', 'humorous', 'dark', 'uplifting', 'suspenseful']
                genre = random.choice(genres)
                theme = random.choice(themes)
                tone = random.choice(tones)
                prompt = self.generate_prompt(genre, theme, tone)
                print(f"\nRandom Prompt: {prompt}\n")
            elif choice == "3":
                print("Happy writing!")
                break
            else:
                print("Invalid choice. Try again.")

if __name__ == "__main__":
    generator = PromptGenerator()
    generator.interactive_session()