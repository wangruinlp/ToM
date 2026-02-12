import os
import sys
import json
import jsonlines
import wandb
import argparse
import scipy.spatial
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)
from torch.nn import CrossEntropyLoss

from datasets import load_dataset
from typing import Dict, List

project_path = os.environ['PROJECT_PATH']
sys.path.append(project_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def word_frequency_statistics(filepath):
    data_vocab = {}
    total_token = 0
    with open(filepath, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        for i in range(len(json_data)):
            text = json_data[i]["text"].split(' ')
            for word in text:
                total_token += 1
                if word in data_vocab:
                    data_vocab[word] += 1
                else:
                    data_vocab[word] = 0

    data_vocab = sorted(data_vocab.items(),  key=lambda d: d[1], reverse=True)
    with open("test.txt", "a") as f:
        f.write(f"Word Total: {total_token}\nWord Type: {len(data_vocab)}\n--------\nWord    Frequency\n")
        for word in data_vocab:
            f.write(f"{word[0]}     {word[1]}\n")



class InitializeModel:
    def __init__(self):
        pass

    def print_model_parameters(self, model):
        print("Layer Name & Parameters")
        print("----------------------------")
        total_params = 0
        for name, parameter in model.named_parameters():
            param_size = parameter.size()
            param_count = torch.prod(torch.tensor(param_size)).item()
            total_params += param_count
            print(f"{name:50} | Size: {str(param_size):30} | Count: {str(param_count):20}")
        print("----------------------------")
        print(f"Total Parameters: {total_params} ({total_params / 1000000:.1f} M)")

    def kaiming_initialization(self, model):
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                torch.nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='leaky_relu')
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
        return model

    def init_model(self, args):
        hidden_size = args.hidden_size
        intermediate_size = (int(hidden_size * 8 / 3 / 128) + 1) * 128
        # intermediate_size = 1024

        config = AutoConfig.for_model(
            model_type=args.backbone,
            hidden_size=args.hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=args.head_num,
            num_hidden_layers=args.layer_num,
            num_key_value_heads=args.head_num,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "/data/yimingwang/llama2_ft/model-hub/Llama-2-7b-chat-hf",
            trust_remote_code=True
        )
        tokenizer.pad_token_id = 0
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float32
        ).to(device)
        model = self.kaiming_initialization(model)

        self.print_model_parameters(model)
        return model, tokenizer





class InitializeDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def process_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        max_token = 256

        encoded_texts = self.tokenizer(examples['text'], add_special_tokens=False)
        input_ids_list = encoded_texts['input_ids']

        new_input_ids_list, new_attn_mask_list = [], []
        for input_ids in input_ids_list:
            temp = input_ids[-max_token + 1:] + [self.tokenizer.eos_token_id]
            new_input_ids_list.append(temp)
            new_attn_mask_list.append([1] * len(temp))
        return {
            "input_ids": new_input_ids_list,
            "attention_mask": new_attn_mask_list
        }


    def init_dataset(self, args):

        training_data = []
        dataset_path = os.path.join(project_path, 'Data', f'{args.n}_node_{args.m}_edge_{args.y}_container.jsonl')
        with open(dataset_path) as file:
            for sample in jsonlines.Reader(file):
                if sample["graph_id"] >= 112:
                    continue
                training_data.append(
                    {
                        "text": "Story: \n" + sample["story"] + "Question: " + sample["query"][f"{args.k}-order"]["question"] + "\nAnswer: " + sample["query"][f"{args.k}-order"]["answer"]
                    }
                )
        with open(os.path.join(project_path, 'Data', f'{args.n}_node_{args.m}_edge_{args.y}_container_train.json'), 'w', encoding='utf-8') as fw:
            json.dump(training_data, fw, indent=2, ensure_ascii=False)


        dataset_path = os.path.join(project_path, 'Data', f'{args.n}_node_{args.m}_edge_{args.y}_container_train.json')
        pt_train = load_dataset('json', data_files=dataset_path, split='train').shuffle()
        pt_val = load_dataset('json', data_files=dataset_path, split='train[:5%]').shuffle()
        pt_train = pt_train.map(
            self.process_func,
            batched=True,
            num_proc=8,
            remove_columns=pt_train.column_names
        )
        pt_val = pt_val.map(
            self.process_func,
            batched=True,
            num_proc=8,
            remove_columns=pt_val.column_names
        )

        return pt_train, pt_val


def train(args):
    wandb.init(
        project="pretrain-tom",
        config={
            # "learning_rate": 0.02,
            "architecture": "llama",
            "dataset": "tom",
            "epochs": 50,
            # "steps": 5000
        }
    )

    model, tokenizer = InitializeModel().init_model(args)
    pt_train, pt_val = InitializeDataset(tokenizer).init_dataset(args)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    ckpt_dir = f'/amax/yimingwang/Interpreting_ToM/Checkpoints/layer{args.layer_num}_hidden{args.hidden_size}_head{args.head_num}_n{args.n}m{args.m}x{args.x}y{args.y}k{args.k}'
    training_arguments = TrainingArguments(
        output_dir=ckpt_dir,
        per_device_train_batch_size=128,
        optim="adamw_torch",
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
        # save_steps=200,
        logging_steps=20,
        group_by_length=False,
        # max_steps=10000,
        num_train_epochs=50,
        gradient_checkpointing=True,
        # max_grad_norm=0.3,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_steps=200,
        weight_decay=1e-3,
    )

    trainer = Trainer(
        model=model,
        train_dataset=pt_train,
        eval_dataset=pt_val,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=data_collator
    )

    output_dir = project_path + '/Training_output/' + args.backbone
    trainer.train()
    trainer.model.save_pretrained(output_dir)

    wandb.finish()


def inference(args):
    model, tokenizer = InitializeModel().init_model(args)
    model = model.from_pretrained(f'/amax/yimingwang/Interpreting_ToM/Checkpoints/layer{args.layer_num}_hidden{args.hidden_size}_head{args.head_num}_n{args.n}m{args.m}x{args.x}y{args.y}k{args.k}/checkpoint-75400')
    model.to(device)
    model.eval()

    dataset_path = os.path.join(project_path, 'Data', f'{args.infer_n}_node_{args.infer_m}_edge_{args.y}_container.jsonl')
    file = jsonlines.Reader(open(dataset_path))

    right = 0
    cnt = 0
    for sample in file:
        if args.test == "train":
            if sample["graph_id"] >= 112:
                continue
            # if sample["character_id"] > 1 or sample["action_id"] > 1:
            #     continue

        elif args.test == "test":
            if sample["graph_id"] < 112:
                continue
            # if sample["character_id"] > 8 or sample["action_id"] > 8:
            #     continue

        input_text = "Story: \n" + sample["story"] + "Question: " + sample["query"][f"{args.infer_k}-order"]["question"] + "\nAnswer: "
        # print(f'{input_text}\n-----\n')
        max_new_tokens = 32

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        ### output token
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            # do_sample=True,
            # top_k=40,
            # top_p=0.95,
            # temperature=0.8,
            return_dict_in_generate=True,
            output_scores=True,
        )

        output_len = min(len(outputs.scores), max_new_tokens)
        output_seq = tokenizer.decode(outputs.sequences[0][-output_len:], skip_special_tokens=True)
        print(str(cnt) + "\n")
        print(f'LM output: \n{output_seq}\n')
        print(f'ground truth: \n{sample["query"]["1-order"]["answer"]}\n-----\n')

        if sample["query"]["1-order"]["answer"] in output_seq:
            right += 1
        cnt += 1

    print(right, cnt, round(right / cnt * 100, 1))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="pretrain")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"])
    parser.add_argument("--test", type=str, default="train", choices=["train", "test"])

    parser.add_argument("--backbone", type=str, default="llama", choices=["llama", "gpt"])
    parser.add_argument("--layer_num", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--head_num", type=int)

    parser.add_argument("--n", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--x", type=int, default=1)
    parser.add_argument("--y", type=int, default=3)
    parser.add_argument("--k", type=int)

    parser.add_argument("--infer_n", type=int)
    parser.add_argument("--infer_m", type=int)
    parser.add_argument("--infer_x", type=int, default=1)
    parser.add_argument("--infer_y", type=int, default=3)
    parser.add_argument("--infer_k", type=int)



    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)

