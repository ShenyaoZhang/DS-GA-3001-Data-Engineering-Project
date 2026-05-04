import pandas as pd
import torch
import os
import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer


class Labeling:
    def __init__(
        self,
        label_model="qwen",
        target_class="joy",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        few_shot_path=None
    ):
        self.label_model = label_model
        self.target_class = target_class
        self.model_id = model_id
        self.few_shot_path = few_shot_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.few_shot_examples = self.load_few_shot_examples()

    def load_few_shot_examples(self):
        if self.few_shot_path and os.path.exists(self.few_shot_path):
            with open(self.few_shot_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _clean_for_prompt(self, text, max_chars=200):
        text = str(text)
        text = re.sub(r'\S+@\S+', ' ', text)
        text = re.sub(r'\b\w+!\w+(?:!\w+)*\b', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]

    def _build_examples_text(self):
        if not self.few_shot_examples:
            return ""

        selected = self.few_shot_examples[:8]

        lines = []
        for ex in selected:
            short_text = self._clean_for_prompt(ex["text"], max_chars=140)
            lines.append(f"Document: {short_text}")
            lines.append(f"Label: {ex['label']}")
            lines.append("")
        return "\n".join(lines).strip()

    def generate_prompt(self, title):
        if self.label_model == "qwen":
            return self.generate_prompt_qwen(title)
        elif self.label_model == "file":
            return None
        else:
            raise ValueError("Only 'qwen' or 'file' labeling is supported in this notebook.")

    def _base_prompt(self, title: str) -> str:
        examples = self._build_examples_text()
        short_title = self._clean_for_prompt(title, max_chars=200)

        prompt = f'''Task: classify a text as expressing joy or not.

Label 1:
texts expressing happiness, delight, excitement, elation, contentment, gratitude, pleasure, cheerfulness, bliss, or any primarily joyful positive emotion.

Label 0:
texts expressing sadness, anger, fear, surprise, disgust, love (non-joyful), or neutral/ambiguous emotions that are NOT primarily joyful.

Important:
- subtle happiness still counts as 1 if the dominant emotion is clearly joyful
- texts about love or anticipation = 0 unless explicitly expressing joy
- texts expressing relief or pride count as 1 if the overall tone is joyful
- if unsure, output 0

Return only 1 or 0.
'''
        if examples:
            prompt += "\nExamples:\n" + examples + "\n\n"

        prompt += f"Document: {short_title}\nLabel:"
        return prompt

    def generate_prompt_qwen(self, title: str) -> str:
        return self._base_prompt(title.strip())

    def set_model(self):
        if self.label_model == "qwen":
            checkpoint = self.model_id
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Qwen model loaded")
        elif self.label_model == "file":
            self.model = None
            self.tokenizer = None
        else:
            raise ValueError("Only 'qwen' or 'file' labeling is supported in this notebook.")

    def predict_animal_product(self, row):
        label = Labeling.check_already_label(row)
        if label is not None:
            return label
        if self.label_model == "qwen":
            return self.get_qwen_label(row)
        elif self.label_model == "file":
            return self.get_file_label(row)
        else:
            raise ValueError("No supported label model selected")

    def generate_inference_data(self, data, column):
        if self.label_model != "file":
            examples = []
            for _, data_point in data.iterrows():
                raw_text = data_point[column]
                examples.append(
                    {
                        "id": data_point["id"],
                        "title": data_point["title"],
                        "training_text": data_point["clean_title"] if "clean_title" in data_point.index else data_point["title"],
                        "true_label": data_point["label"] if "label" in data_point.index else None,
                        "text": self.generate_prompt(raw_text),
                    }
                )
            data = pd.DataFrame(examples)
        return data

    def _extract_label(self, response_text: str) -> str:
        response_text = str(response_text).strip()

        match = re.search(r'\b([01])\b', response_text)
        if match:
            return match.group(1)

        lowered = response_text.lower()
        if lowered.startswith("1") or "label: 1" in lowered or "label 1" in lowered:
            return "1"
        if lowered.startswith("0") or "label: 0" in lowered or "label 0" in lowered:
            return "0"

        return "0"

    def get_qwen_label(self, row):
        user_prompt = row["text"]

        messages = [
            {
                "role": "system",
                "content": "You are a classifier. Reply with exactly one character: 1 or 0."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=768
        ).to(self.model.device)

        input_len = model_inputs.input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = outputs[:, input_len:]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._extract_label(response)

    def get_file_label(self, row):
        raise NotImplementedError("File-based labeling is not implemented in this notebook.")

    @staticmethod
    def check_already_label(row):
        return None
