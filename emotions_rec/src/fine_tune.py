from typing import Any, Optional, Dict

from transformers import Trainer, TrainingArguments, BertTokenizer, DataCollatorWithPadding, BertForSequenceClassification, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset, DatasetDict
import torch
import pandas as pd
import numpy as np
from torch import nn


class BertFineTuner:
    def __init__(
        self,
        model_name: Optional[str],
        training_data: Optional[pd.DataFrame],
        test_data: Optional[pd.DataFrame],
        learning_rate=2e-5,
        dropout=0.2,
        confidence_threshold=0.85,
        num_labels=6,
        max_length=512,
        num_train_epochs=5,
        monitor_metric="eval_f1",
        batch_size=16,
        results_dir: str = "results",
        log_dir: str = "log",
    ):
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
        self.confidence_threshold = confidence_threshold
        self.num_labels = num_labels
        self.max_length = max_length
        self.num_train_epochs = num_train_epochs
        self.monitor_metric = monitor_metric
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.log_dir = log_dir

        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
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
                max_length=self.max_length
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
                max_length=self.max_length
            )

        tokenized_data = dataset.map(tokenize_function, batched=True)
        return tokenized_data

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        out = {
            "accuracy": accuracy_score(labels, preds),
            "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        }
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 2:
            out["precision_pos"] = precision_score(labels, preds, average="binary", pos_label=1, zero_division=0)
            out["recall_pos"] = recall_score(labels, preds, average="binary", pos_label=1, zero_division=0)
            out["f1_pos"] = f1_score(labels, preds, average="binary", pos_label=1, zero_division=0)
        return out

    def train_data(self, df, still_unbalenced):
        early_stopping_callback = EarlyStoppingCallback(patience=5, monitor=self.monitor_metric, mode="max", log_dir=self.log_dir)
        tokenized_data, data_collator = self.create_dataset(df, self.test_data)

        # Compute dynamic class weights from label distribution
        class_weights = None
        if still_unbalenced:
            label_counts = df["label"].value_counts().sort_index()
            total = len(df)
            class_weights = torch.tensor(
                [total / (self.num_labels * label_counts.get(i, 1)) for i in range(self.num_labels)],
                dtype=torch.float32
            )
            class_weights = class_weights / class_weights.sum() * self.num_labels
            print(f"Dynamic class weights: {class_weights.tolist()}")

        training_args = TrainingArguments(
            output_dir=self.results_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model=self.monitor_metric,
            greater_is_better=True,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            save_total_limit=2,
            logging_steps=10,
            push_to_hub=False,
            load_best_model_at_end=True,
            report_to=[]
        )

        trainer_kwargs = dict(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=BertFineTuner.compute_metrics,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["val"],
            callbacks=[early_stopping_callback]
        )

        if still_unbalenced:
            trainer = MyTrainer(class_weights=class_weights, **trainer_kwargs)
        else:
            trainer = Trainer(**trainer_kwargs)

        trainer.train()
        print("Best checkpoint:", trainer.state.best_model_checkpoint)

        results = trainer.evaluate()
        print(results)

        self.trainer = trainer
        self.model = trainer.model
        return results, self.trainer

    def get_inference(self, df: pd.DataFrame) -> torch.Tensor:
        """Return predicted class labels (0-5) via argmax."""
        predicted_labels = []
        chunk_size = 10000
        total_records = len(df)
        start_index = 0

        while start_index < total_records:
            end_index = min(start_index + chunk_size, total_records)
            chunk = df[start_index:end_index]
            test_dataset = self.create_test_dataset(chunk)
            predictions = self.trainer.predict(test_dataset["test"])
            batch_labels = torch.argmax(torch.tensor(predictions.predictions), dim=1)
            predicted_labels.append(batch_labels)
            start_index = end_index

        return torch.cat(predicted_labels)

    def get_inference_with_probs(self, df: pd.DataFrame):
        """Return (predicted_labels, confidence_scores) tensors."""
        predicted_labels = []
        confidences = []
        chunk_size = 10000
        total_records = len(df)
        start_index = 0

        while start_index < total_records:
            end_index = min(start_index + chunk_size, total_records)
            chunk = df[start_index:end_index]
            test_dataset = self.create_test_dataset(chunk)
            predictions = self.trainer.predict(test_dataset["test"])
            probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)
            batch_labels = torch.argmax(probs, dim=1)
            batch_confidence = probs.max(dim=1).values
            predicted_labels.append(batch_labels)
            confidences.append(batch_confidence)
            start_index = end_index

        return torch.cat(predicted_labels), torch.cat(confidences)

    def save_model(self, path: str):
        if self.trainer is not None:
            self.trainer.save_model(path)

    def update_model(self, model_name, model_acc, save_model: bool):
        if save_model and self.trainer is not None:
            self.save_model(model_name)

        self.last_model_acc = {model_name: model_acc}
        self.base_model = model_name


class MyTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            weight = self.class_weights.to(model.device)
        else:
            weight = None
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, monitor="eval_f1", mode="max", log_dir=None):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.wait = 0
        self.log_dir = log_dir

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.is_world_process_zero and state.log_history:
            current_value = None
            for log_entry in reversed(state.log_history):
                if self.monitor in log_entry:
                    current_value = log_entry[self.monitor]
                    break

            if current_value is not None:
                improved = (current_value > self.best_value) if self.mode == "max" else (current_value < self.best_value)
                if improved:
                    self.best_value = current_value
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        control.should_training_stop = True

                if self.log_dir:
                    with open(f"{self.log_dir}/epoch_{state.epoch}.txt", "w") as f:
                        for log in state.log_history:
                            f.write(f"{log}\n")
