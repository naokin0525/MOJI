import argparse
import tkinter as tk
from tkinter import ttk
from generate import generate_handwriting, HandwritingModel
import torch

class HandwritingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Generator")
        
        tk.Label(root, text="Text:").pack()
        self.text_entry = tk.Entry(root, width=50)
        self.text_entry.pack()
        
        tk.Label(root, text="Model Path:").pack()
        self.model_path = tk.Entry(root, width=50)
        self.model_path.pack()
        
        tk.Label(root, text="Style:").pack()
        self.style = ttk.Combobox(root, values=["casual", "formal", "cursive"])
        self.style.set("casual")
        self.style.pack()
        
        self.variation = tk.BooleanVar()
        tk.Checkbutton(root, text="Random Variation", variable=self.variation).pack()
        
        tk.Label(root, text="Stroke Width:").pack()
        self.stroke_width = tk.Entry(root, width=10)
        self.stroke_width.insert(0, "1.2")
        self.stroke_width.pack()
        
        tk.Button(root, text="Generate", command=self.generate).pack()

    def generate(self):
        args = argparse.Namespace(
            model_path=self.model_path.get(),
            text=self.text_entry.get(),
            output="output/output.svg",
            random_variation=self.variation.get(),
            stroke_width=float(self.stroke_width.get()),
            style=self.style.get(),
            convert_to=None
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        char_to_id = torch.load(f"{args.model_path}/char_to_id.pt")
        model = HandwritingModel(char_vocab_size=len(char_to_id), embedding_dim=50, hidden_dim=256, latent_dim=64).to(device)
        model.load_state_dict(torch.load(f"{args.model_path}/model.pt"))
        generate_handwriting(model, args.text, char_to_id, device, args)
        tk.Label(self.root, text="Generated at output/output.svg").pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingGUI(root)
    root.mainloop()