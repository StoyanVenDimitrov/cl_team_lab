import os
import json
import torch
from transformers import AlbertTokenizer, AlbertConfig, AlbertForSequenceClassification
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from tqdm import trange, tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.utils import utils
from pytorch_model_summary import summary
# import wandb
# wandb.init(reinit=True)

MODELS = {
    "albert": [AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification], 
    "bert": [BertConfig, BertTokenizer, BertForSequenceClassification], 
    "scibert": [BertConfig, BertTokenizer, BertForSequenceClassification]
    }

class TransformerModel:
    def __init__(self, args, config):
        self.args = args
        self._config = config
        # if self.args.log_metrics:
        #     self.init_logging(self._config)
        self.pre_config = config["preprocessor"]
        config = config["torch"]
        self.config = config
        self.epochs = int(config["epochs"])
        self.batch_size = int(config["batch_size"])
        self.learning_rate = float(config["learning_rate"])
        self.max_len = int(config["max_len"])
        self.use_bfloat16 = True if config["bfloat16"] == "True" else False
        self.model_type = config["model_type"]
        self.model_version = config["model_version"]
        # if "uncased" in self.model_version:
        #     self.do_lower_case = True
        # elif "cased" in self.model_version:
        #     self.do_lower_case = False
        # else:
        #     self.do_lower_case = True if config["lowercase"] == "True" else False

        self.init_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logdir = utils.make_logdir("torch", "Torch", self.pre_config, config)

    def init_model(self):
        print("Initializing model...")
        configurator, tokenizer, model = MODELS[self.model_type]

        config = configurator.from_pretrained(self.model_version)
        config.num_labels = 3
        config.use_bfloat16 = self.use_bfloat16
        
        self.tokenizer = tokenizer.from_pretrained(self.model_version, config=config)
        # self.tokenizer.do_lower_case = self.do_lower_case"tdev

        self.model = model.from_pretrained(self.model_version, config=config)
        print("Model initialized!")

    def encode(self, text):
        if self.pre_config["lemmatize"] == "True":
            if "lemmatized_string" in text.keys():
                key = "lemmatized_string"
            else:
                raise KeyError(f"You have not lemmatized the data yet. Execute './run_lemmatizer.sh' to lemmatize the data.")
        else:
            key = "string"
        return self.tokenizer(text[key], truncation=True, max_length=self.max_len, padding="max_length")

    def prepare_data(self, train, dev, test, batch_size=8):
        print("Preparing data...")
        train_tokens = train.map(self.encode, batched=True)
        dev_tokens = dev.map(self.encode, batched=True)
        test_tokens = test.map(self.encode, batched=True)

        train_tokens.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        dev_tokens.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        test_tokens.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

        train_dataloader = torch.utils.data.DataLoader(train_tokens, batch_size=batch_size)
        dev_dataloader = torch.utils.data.DataLoader(dev_tokens, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_tokens, batch_size=batch_size)

        print("Data prepared!")

        return train_dataloader, dev_dataloader, test_dataloader

    def train(self, train_dataloader, dev_dataloader):
        print("Starting training...")

        try:
            print(summary(self.model, torch.zeros((self.batch_size, self.max_len), dtype=torch.long), show_input=True))
        except:
            print("Unable to print model summary.")

        self.model.to(self.device)

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
                loss = outputs[0]
                loss.backward()
                losses += loss.item()
                optimizer.step()
                optimizer.zero_grad()

                train_true.append(batch["labels"].tolist())
                train_pred.append(torch.argmax(outputs[1], dim=1).tolist())

            train_y_true = [v for l in train_true for v in l]
            train_y_pred = [v for l in train_pred for v in l]
            train_acc = accuracy_score(train_y_true, train_y_pred)
            train_f1 = f1_score(train_y_true, train_y_pred, average="macro")
            print(f"Mean train loss: {losses / (i + 1)}\t", f"Train acc: {train_acc}", f"Train f1: {train_f1}")

            # Evaluate on validation data
            self.model.eval()

            dev_true = []
            dev_pred = []

            with torch.no_grad():
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

                # if self.args.log_metrics:
                #     utils.wandb_log_report("val", val_report)

        print("Finished training!")

        save_path = os.path.join(self.logdir, os.pardir, "model.pt")
        print("Saving model to", save_path)
        torch.save(self.model.state_dict(), save_path)

    def evaluate(self, dataloader, save_output=True):
        print("Evaluating...")

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

        report_json = classification_report(test_y_true, test_y_pred, labels=[0,1,2], output_dict=True)
        report_text = classification_report(test_y_true, test_y_pred, labels=[0,1,2])
        print(report_text)

        if save_output:
            results_path = os.path.join(self.logdir, os.pardir)
            with open(results_path + "/results.json", "w") as f:
                json.dump(report_json, f)
            with open(results_path + "/results.txt", "w") as f:
                f.write(report_text)
            print("Saved result files results.json and results.txt to:", results_path)

        # if self.args.log_metrics:
        #     utils.wandb_log_report("test", test_report)

    def init_logging(self, config):
        if self.args.log_metrics:
            wandb.init(project="citent-torch", name="citent-torch-"+config["torch"]["model_version"], reinit=True)
            utils.wandb_init(config["torch"], _type="model")
            utils.wandb_init(config["preprocessor"], _type="preprocess")
            utils.wandb_init_metrics()
            print("Init wandb logging...")
