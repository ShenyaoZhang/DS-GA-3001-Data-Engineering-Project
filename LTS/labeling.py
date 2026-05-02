import pandas as pd
from pprint import pprint
import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from openai import OpenAI
import pandas as pd


class Labeling:
    def __init__(self, label_model= "llama"):
        self.label_model = label_model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def generate_prompt(self, title):
        if self.label_model == "llama":
            return self.generate_prompt_llama(title)
        elif self.label_model == "gpt":
            return self.generate_prompt_gpt(title)
        elif self.label_model == "qwen":
            # Same task text as GPT; formatting for the chat template happens in get_qwen_label.
            return self.generate_prompt_gpt(title)
        else:
            return None


    def generate_prompt_llama(self, title: str) -> str:
        return f"""### Instruction: {self.prompt_llama}
                ### Input:
                {title.strip()}
                """

    def generate_prompt_gpt(self, title):
        # # Original wildlife product classification prompt:
        # return f'''You are labeling tool to create labels for a classification task .
        #                      I will provide text data from an advertisement of a product.
        #                      The product should be classified in two labels:
        #                      Label 1: relevant animal - if the product is from any of those 3 animals: Shark, Ray or Chimaeras. It should be from a real animal. Not an image or plastic for example.
        #                      Label 2: not a relevant animal - if the product is from any other animal that is not Shark, Ray, or Chimaeras, or if the product is 100% synthetic (vegan).
        #                      ...
        #                      6. Advertisement: {title}
        #                      Label:
        #                      '''

        # Reuters topic classification prompt (earn):
        return f'''You are a labeling tool to create labels for a text classification task.
I will provide the title (and possibly a short excerpt) of a Reuters news article.
The article should be classified into one of two labels:
Label 1: earn - if the article is about corporate earnings, financial results,
profits, losses, dividends, revenue reports, or quarterly/annual results.
Label 2: not earn - if the article is about any other topic such as trade,
commodities, politics, acquisitions, monetary policy, etc.
Return only one of the two labels: earn or not earn. No explanation is necessary.

Examples:
1. Article: NATIONAL AVERAGE PRICES FOR FARMER-OWNED RESERVE
Label: not earn

2. Article: ALLIED-LYONS YEAR PRETAX PROFIT RISES
Label: earn

3. Article: JAPAN TO REVISE LONG-TERM ENERGY DEMAND DOWNWARDS
Label: not earn

4. Article: SEARS ROEBUCK 1ST-QTR NET INCOME UP
Label: earn

5. Article: {title}
Label:
'''

    def generate_llama_prompt(self):
        # # Original wildlife product classification prompt for LLaMA:
        # f'''You are labeling tool to create labels for a classification task .
        #                      I will provide text data from an advertisement of a product.
        #                      The product should be classified in two labels:
        #                      relevant animal - ...
        #                      not a relevant animal - ...
        #                      6. Advertisement:
        #                      '''

        # Reuters topic classification prompt (earn) for LLaMA:
        return f'''You are a labeling tool for a text classification task.
I will provide the title of a Reuters news article.
Classify the article into one of two labels:
earn - if the article is about corporate earnings, financial results,
profits, losses, dividends, revenue reports, or quarterly/annual results.
not earn - if the article is about any other topic.
Return only one of the two labels: earn or not earn.
Article:
'''

    def set_model(self):
        if self.label_model == "llama":
            checkpoint = "llama/"
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.prompt_llama = self.generate_llama_prompt()
            print("model Loaded")
        elif self.label_model == "gpt":
            self.model = OpenAI(api_key="YOUR_OPENAI_API_KEY")
        elif self.label_model == "qwen":
            # Local dir (e.g. ./qwen) or Hugging Face id, e.g. Qwen/Qwen2.5-1.5B-Instruct
            checkpoint = os.environ.get("QWEN_MODEL_DIR", "Qwen/Qwen2.5-1.5B-Instruct")
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)
            self.model.eval()
            print(f"Qwen loaded from {checkpoint}")
        elif self.label_model =="file":
            self.model = None


    def predict_animal_product(self, row):
        # print(f"Prediction Animal with {self.model}")
        label = Labeling.check_already_label(row)
        if label:
            return label
        if self.label_model == "llama":
            return self.get_llama_label(row)
        elif self.label_model == "gpt":
            return self.get_gpt_label(row)
        elif self.label_model == "qwen":
            return self.get_qwen_label(row)
        elif self.label_model == "file":
            return self.get_file_label(row)
        else:
            raise ValueError("No model selected")


    def generate_inference_data(self, data, column):
        def truncate_string(s, max_length=2000):  # Adjust max_length as needed
            return s[:max_length] + '...' if len(s) > max_length else s

        if self.label_model != "file":
            examples = []
            for _, data_point in data.iterrows():
                examples.append(
                {
                    "id": data_point["id"],
                    "title": data_point["title"],
                    "training_text": data_point["clean_title"],
                    "text": self.generate_prompt(truncate_string(data_point[column])),
                }
                )
            data = pd.DataFrame(examples)
        return data



    def get_gpt_label(self, row):
        if os.path.exists("labaled_by_gpt.csv"):
            labels =  pd.read_csv("labaled_by_gpt.csv")
        else:
            labels = None
        id_ = row["id"]
        prompt = row["text"]
        if labels:
            if id_ in labels["id"].to_list():
                return labels.loc[labels["id"] == id_, "label"].values[0]
        response = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.2,
            )
        return response.choices[0].message.content


    def get_llama_label(self, row):
        text = row["text"]
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        inputs_length = len(inputs["input_ids"][0])
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.0001)
            results = self.tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)
            try:
                answer = results.split("Response:\n")[2].split("\n")[0]
            except Exception:
                # Handle IndexError separately
                try:
                    answer = results.split("Response:\n")[1].split("\n")[0]
                except Exception:
                    # Handle any other exception
                    # answer = 'not a relevant animal'  # Original: wildlife default
                    answer = 'not earn'  # Reuters default
        return answer

    def get_qwen_label(self, row):
        user_content = row["text"]
        messages = [{"role": "user", "content": user_content}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = user_content
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text_out = self.tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip().lower()
        if "not earn" in text_out[:32]:
            return "not earn"
        if text_out.startswith("earn") or text_out[:12].strip() == "earn":
            return "earn"
        return "not earn"

    def get_file_label(self, row):
        # raise NotImplementedError()  # Original: not implemented
        if "label" in row and pd.notna(row["label"]):
            return "earn" if int(row["label"]) == 1 else "not earn"
        if "label_earn" in row and pd.notna(row["label_earn"]):
            return "earn" if int(row["label_earn"]) == 1 else "not earn"
        return "not earn"

    @staticmethod
    def check_already_label(row):
        return None
        # labeled_data = pd.read_csv("all_labeled_data_gpt.csv")
        # if row["id"] in labeled_data["id"].values:
        #     # Retrieve the label for the corresponding id
        #     print("data already labeled")
        #     label = labeled_data.loc[labeled_data["id"] == row["id"], "label"].values[0]
        #     return label
        # else:
        #     return None


