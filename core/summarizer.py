# core/llm/summarizer.py
from typing import List, Dict
import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration

class DocumentSummarizer:
    def __init__(self):
        self.model_name = settings.MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.max_length = 1024
        self.min_length = 50
        self._initialized = False

    async def initialize(self):
        if not self._initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqGeneration.from_pretrained(self.model_name)
            
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                
            self._initialized = True

    async def summarize_documents(self, documents: List[Dict]) -> str:
        texts = [doc.get("content", "") for doc in documents]
        combined_text = "\n\n".join(texts)
        
        inputs = self.tokenizer(combined_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            min_length=self.min_length,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
