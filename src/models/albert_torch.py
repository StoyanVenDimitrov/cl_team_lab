import os
import torch
from transformers import AlbertTokenizer, AlbertConfig, AlbertForSequenceClassification
from tqdm import trange, tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.utils.utils import wandb_log_report

from src.utils.utils import make_filename


class AlbertModel:
    def __init__(self, config):
        self.config = config
        self.epochs = int(config["epochs"])
        self.learning_rate = float(config["learning_rate"])
        self.max_len = int(config["max_len"])

        version = config["albert_version"]
        model_config = AlbertConfig.from_pretrained(version)
        model_config.num_labels = 3
        model_config.use_bfloat16 = True if config["bffloat16"] == "True" else False
        self.tokenizer = AlbertTokenizer.from_pretrained(version, config=model_config)
        self.model = AlbertForSequenceClassification.from_pretrained(version, config=model_config)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def encode(self, text):
        return self.tokenizer(text["string"], truncation=True, max_length=self.max_len, padding="max_length")

    def prepare_data(self, train, dev, test, batch_size=8):
        train_tokens = train.map(self.encode, batched=True)
        dev_tokens = dev.map(self.encode, batched=True)
        test_tokens = test.map(self.encode, batched=True)

        train_tokens.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        dev_tokens.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        test_tokens.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

        train_dataloader = torch.utils.data.DataLoader(train_tokens, batch_size=batch_size)
        dev_dataloader = torch.utils.data.DataLoader(dev_tokens, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_tokens, batch_size=batch_size)

        return train_dataloader, dev_dataloader, test_dataloader

    def train(self, train_dataloader, dev_dataloader):

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.learning_rate)

        for epoch in trange(self.epochs, desc="Epoch 1"):
            self.model.train()

            train_true = []
            train_pred = []
            losses = 0.0

            for i, batch in enumerate(tqdm(train_dataloader)):
                batch["labels"] = batch.pop("label")
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                #         print(torch.argmax(outputs[1], dim=1))
                loss = outputs[0]
                loss.backward()
                losses += loss.item()
                #         loss.sum().backward()  # multiple GPUs
                #         loss.backward(torch.Tensor([1,1]))
                optimizer.step()
                optimizer.zero_grad()

                train_true.append(batch["labels"].tolist())
                train_pred.append(torch.argmax(outputs[1], dim=1).tolist())

            train_y_true = [v for l in train_true for v in l]
            train_y_pred = [v for l in train_pred for v in l]
            train_acc = accuracy_score(train_y_true, train_y_pred)
            train_f1 = f1_score(train_y_true, train_y_pred, average="macro")
            print(f"Mean train loss: {losses / (i + 1)}\t", f"Train acc: {train_acc}", f"Train f1: {train_f1}")

            self.model.eval()
            with torch.no_grad():
                dev_true = []
                dev_pred = []
                for i, dev_batch in enumerate(tqdm(dev_dataloader)):
                    dev_batch["labels"] = dev_batch.pop("label")
                    dev_batch = {k: v.to(self.device) for k, v in dev_batch.items()}
                    dev_outputs = self.model(**dev_batch)
                    dev_true.append(dev_batch["labels"].detach().cpu().tolist())
                    dev_pred.append(torch.argmax(dev_outputs[1].detach().cpu(), dim=1).tolist())
                dev_y_true = [v for l in dev_true for v in l]
                dev_y_pred = [v for l in dev_pred for v in l]
                dev_acc = accuracy_score(dev_y_true, dev_y_pred)
                dev_f1 = f1_score(dev_y_true, dev_y_pred, average="macro")
                print(f"Dev accuracy: \t{dev_acc}", f"Dev f1: \t{dev_f1}")

                val_report = classification_report(dev_y_true, dev_y_pred, labels=[0, 1, 2], output_dict=True)

                if self.args.log_metrics:
                    wandb_log_report("val", val_report)

        torch.save(self.model.state_dict(), os.path.join("saved_models", make_filename(self.config)))

    def evaluate(self, dataloader, print_report=True):

        self.model.eval()

        test_true = []
        test_pred = []

        with torch.no_grad():
            for i, test_batch in enumerate(tqdm(dataloader)):
                test_batch["labels"] = test_batch.pop("label")
                test_batch = {k: v.to(self.device) for k, v in test_batch.items()}
                test_outputs = self.model(**test_batch)
                test_true.append(test_batch["labels"].detach().cpu().tolist())
                test_pred.append(torch.argmax(test_outputs[1].detach().cpu(), dim=1).tolist())
            test_y_true = [v for l in test_true for v in l]
            test_y_pred = [v for l in test_pred for v in l]
            test_f1 = f1_score(test_y_true, test_y_pred, average="macro")
            print(f"Test f1: \t{test_f1}")

        test_report = classification_report(test_y_true, test_y_pred, labels=[0,1,2], output_dict=True)
        if print_report:
            print(classification_report(test_y_true, test_y_pred, labels=[0,1,2]))

        if self.args.log_metrics:
            wandb_log_report("test", test_report)


