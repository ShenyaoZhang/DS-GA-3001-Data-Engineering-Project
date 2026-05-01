
from typing import Any, Optional, Dict

from transformers import Trainer, TrainingArguments, BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
import torch
import pandas as pd
from torch import nn


class BertFineTuner:
    def __init__(self, model_name: Optional[str], training_data: Optional[pd.DataFrame], test_data: Optional[pd.DataFrame], learning_rate=2e-5, dropout=0.2):
        self.base_model = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.last_model_acc: Dict[str, float] = None
        self.training_data = training_data
        self.test_data = test_data
        self.trainer = None
        self.run_clf = False
        self.learning_rate = learning_rate
        self.weight_decay = 0.01

        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        if dropout:
            model.config.hidden_dropout_prob = dropout
            model.config.attention_probs_dropout_prob = dropout

        self.model = model
        self.model.to(self.device)

    def set_clf(self, set_value: bool):
        self.run_clf = set_value

    def get_clf(self):
        return self.run_clf

    def get_last_model_acc(self):
        return self.last_model_acc

    def get_base_model(self):
        return self.base_model

    def _text_col(self, df):
        if "training_text" in df.columns:
            return "training_text"
        return "title"

    def create_dataset(self, train, test):
        text_col_train = self._text_col(train)
        text_col_test = self._text_col(test)

        dataset_train = Dataset.from_pandas(
            train[[text_col_train, "label"]].rename(columns={text_col_train: "text"})
        )
        dataset_val = Dataset.from_pandas(
            test[[text_col_test, "label"]].rename(columns={text_col_test: "text"})
        )

        dataset = DatasetDict()
        dataset["train"] = dataset_train
        dataset["val"] = dataset_val

        def tokenize_function(element):
            return self.tokenizer(
                element["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        tokenized_data = dataset.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        return tokenized_data, data_collator

    def create_test_dataset(self, df: pd.DataFrame) -> Dataset:
        text_col = self._text_col(df)
        test_dataset = Dataset.from_pandas(
            df[[text_col]].rename(columns={text_col: "text"})
        )

        dataset = DatasetDict()
        dataset["test"] = test_dataset

        def tokenize_function(element):
            return self.tokenizer(
                element["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )

        tokenized_data = dataset.map(tokenize_function, batched=True)
        return tokenized_data

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return {
            "accuracy": accuracy_score(labels, preds),

            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),

            "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),

            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),

            "precision_pos": precision_score(labels, preds, average="binary", pos_label=1, zero_division=0),
            "recall_pos": recall_score(labels, preds, average="binary", pos_label=1, zero_division=0),
            "f1_pos": f1_score(labels, preds, average="binary", pos_label=1, zero_division=0),
        }

    def train_data(self, df, still_unbalenced):
        early_stopping_callback = EarlyStoppingCallback(patience=5, log_dir="./log")
        tokenized_data, data_collator = self.create_dataset(df, self.test_data)

        training_args = TrainingArguments(
            output_dir="results",
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_accuracy",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            save_total_limit=2,
            logging_steps=10,
            push_to_hub=False,
            load_best_model_at_end=False,
            report_to=[]
        )

        trainer_class = MyTrainer if still_unbalenced else Trainer
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=BertFineTuner.compute_metrics,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["val"],
            callbacks=[early_stopping_callback]
        )

        trainer.train()
        print("Best checkpoint:", trainer.state.best_model_checkpoint)

        results = trainer.evaluate()
        print(results)

        self.trainer = trainer
        self.model = trainer.model
        return results, self.trainer

    def get_inference(self, df: pd.DataFrame) -> torch.Tensor:
        predicted_labels = []
        chunk_size = 10000
        total_records = len(df)
        start_index = 0

        while start_index < total_records:
            end_index = min(start_index + chunk_size, total_records)
            chunk = df[start_index:end_index]
            test_dataset = self.create_test_dataset(chunk)
            predictions = self.trainer.predict(test_dataset["test"])
            prediction_scores = predictions.predictions
            batch_predicted_labels = torch.argmax(torch.tensor(prediction_scores), dim=1)
            predicted_labels.append(batch_predicted_labels)
            start_index = end_index

        return torch.cat(predicted_labels)

    def save_model(self, path: str):
        if self.trainer is not None:
            self.trainer.save_model(path)

    def update_model(self, model_name, model_acc, save_model: bool):
        if save_model and self.trainer is not None:
            self.save_model(model_name)

        self.last_model_acc = {model_name: model_acc}
        self.base_model = model_name
        # Do not reload checkpoint here.


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, log_dir=None):
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0
        self.log_dir = log_dir

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and state.log_history:
            current_loss = None
            for log_entry in reversed(state.log_history):
                if "eval_loss" in log_entry:
                    current_loss = log_entry["eval_loss"]
                    break

            if current_loss is not None:
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        control.should_training_stop = True

                if self.log_dir:
                    with open(f"{self.log_dir}/epoch_{state.epoch}.txt", "w") as f:
                        for log in state.log_history:
                            f.write(f"{log}\n")
