import pandas as pd
from pprint import pprint
import torch
import os
import json
from typing import List, Tuple, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from openai import OpenAI
import pandas as pd


# Default binary task: Reuters "earn" vs "not earn".
# Override per-run via Labeling(...) kwargs or env vars LTS_POSITIVE_LABEL /
# LTS_NEGATIVE_LABEL / LTS_TASK_DESCRIPTION / LTS_EXAMPLES_JSON so the same
# pipeline can serve other binary triage tasks (e.g. "trade", "acq", or the
# original wildlife-product task) without code edits.
DEFAULT_POSITIVE_LABEL = "earn"
DEFAULT_NEGATIVE_LABEL = "not earn"
DEFAULT_TASK_DESCRIPTION = (
    "the title (and possibly a short excerpt) of a Reuters news article. "
    "Label 1 ({pos}) if the article is about corporate earnings, financial "
    "results, profits, losses, dividends, revenue reports, or quarterly/annual "
    "results. Label 2 ({neg}) for any other topic such as trade, commodities, "
    "politics, acquisitions, monetary policy, etc."
)
DEFAULT_EXAMPLES: List[Tuple[str, str]] = [
    ("NATIONAL AVERAGE PRICES FOR FARMER-OWNED RESERVE", "not earn"),
    ("ALLIED-LYONS YEAR PRETAX PROFIT RISES", "earn"),
    ("JAPAN TO REVISE LONG-TERM ENERGY DEMAND DOWNWARDS", "not earn"),
    ("SEARS ROEBUCK 1ST-QTR NET INCOME UP", "earn"),
]


def _load_examples_from_env() -> Optional[List[Tuple[str, str]]]:
    path = os.environ.get("LTS_EXAMPLES_JSON")
    if not path:
        return None
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return [(str(item["title"]), str(item["label"])) for item in raw]
    except Exception as exc:
        print(f"[Labeling] failed to load LTS_EXAMPLES_JSON='{path}': {exc}")
        return None


class Labeling:
    def __init__(
        self,
        label_model: str = "llama",
        positive_label: Optional[str] = None,
        negative_label: Optional[str] = None,
        task_description: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
    ):
        self.label_model = label_model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.positive_label = (
            positive_label
            or os.environ.get("LTS_POSITIVE_LABEL")
            or DEFAULT_POSITIVE_LABEL
        ).strip()
        self.negative_label = (
            negative_label
            or os.environ.get("LTS_NEGATIVE_LABEL")
            or DEFAULT_NEGATIVE_LABEL
        ).strip()
        raw_desc = (
            task_description
            or os.environ.get("LTS_TASK_DESCRIPTION")
            or DEFAULT_TASK_DESCRIPTION
        )
        self.task_description = raw_desc.format(
            pos=self.positive_label, neg=self.negative_label
        )
        self.examples = (
            examples
            or _load_examples_from_env()
            or DEFAULT_EXAMPLES
        )

    @property
    def pos(self) -> str:
        return self.positive_label.lower()

    @property
    def neg(self) -> str:
        return self.negative_label.lower()

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

    def _examples_block(self) -> str:
        lines = []
        for idx, (ex_title, ex_label) in enumerate(self.examples, start=1):
            lines.append(f"{idx}. Article: {ex_title}\nLabel: {ex_label}\n")
        return "\n".join(lines)

    def generate_prompt_gpt(self, title):
        next_idx = len(self.examples) + 1
        return (
            "You are a labeling tool to create labels for a text classification task.\n"
            f"I will provide {self.task_description}\n"
            f"Return only one of the two labels: {self.positive_label} or "
            f"{self.negative_label}. No explanation is necessary.\n\n"
            "Examples:\n"
            f"{self._examples_block()}\n"
            f"{next_idx}. Article: {title}\nLabel:\n"
        )

    def generate_llama_prompt(self):
        return (
            "You are a labeling tool for a text classification task.\n"
            f"I will provide {self.task_description}\n"
            f"Return only one of the two labels: {self.positive_label} or "
            f"{self.negative_label}.\nArticle:\n"
        )

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
                    answer = self.negative_label
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
        # Match the more specific (longer) label first so 'not earn' wins over 'earn'
        # when the model emits 'not earn'.
        labels = sorted([self.pos, self.neg], key=len, reverse=True)
        for lab in labels:
            if lab and lab in text_out[:64]:
                return lab
        return self.neg

    def get_file_label(self, row):
        if "label" in row and pd.notna(row["label"]):
            return self.pos if int(row["label"]) == 1 else self.neg
        # Backward compat: prepare_reuters previously emitted label_<topic> columns.
        topic_col = f"label_{self.pos.replace(' ', '_')}"
        if topic_col in row and pd.notna(row[topic_col]):
            return self.pos if int(row[topic_col]) == 1 else self.neg
        return self.neg

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


