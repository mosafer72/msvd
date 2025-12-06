# ======================= madv.py (FIXED & CLEAN VERSION) =======================
from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import math
import numpy as np
import torch
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import models
import utils
from utils import multi_data_loader_st, data_loader
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)


# ======================= Dataset Loader =======================
def load_dataset(tokenizer, file_path, args):
    data_dir = args.data_dir
    df = pd.read_csv(os.path.join(data_dir, file_path))

    funcs = df["func"].tolist()
    labels = df["target"].tolist()

    inputs, classes = [], []

    for func, label in tqdm(zip(funcs, labels), total=len(funcs)):
        tokens, ids = convert_examples_to_features(func, tokenizer, args)
        inputs.append(ids)
        classes.append(label)

    return np.array(inputs), np.array(classes)


def convert_examples_to_features(code, tokenizer, args):

    code_tokens = tokenizer.tokenize(str(code))[:args.block_size - 2]

    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return source_tokens, source_ids


# ======================= Seed =======================
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# ======================= Training =======================
def train(model, tokenizer, args):

    source_files = args.source_train_file_list
    valid_files = args.source_valid_file_list

    source_inputs, source_labels = [], []
    source_sizes = []

    # Load Source Domains
    for sf in source_files:
        x, y = load_dataset(tokenizer, sf, args)
        source_inputs.append(x)
        source_labels.append(y)
        source_sizes.append(len(x))

    # Load Target Train
    target_train_inputs, _ = load_dataset(tokenizer, args.target_train_file, args)

    # Load Validation Domains (as target test)
    valid_inputs, valid_labels = [], []
    for vf in valid_files:
        vx, vy = load_dataset(tokenizer, vf, args)
        valid_inputs.append(vx)
        valid_labels.append(vy)

    target_test_inputs = np.concatenate(valid_inputs)
    target_test_labels = np.concatenate(valid_labels)

    # Prepare Training Steps
    n_batch = int(min(source_sizes) / args.train_batch_size)
    args.max_steps = args.epochs * n_batch
    args.warmup_steps = args.max_steps // 20

    optimizer = torch.optim.AdamW(model.get_parameters(args, lr=args.learning_rate),
                                  lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    # Logging
    number_domain = args.number_domain
    logger.info("***** TRAINING *****")
    for i in range(number_domain):
        logger.info(f"Source Domain {i+1}: {len(source_inputs[i])} samples")
    logger.info(f"Target Train Samples: {len(target_train_inputs)}")
    logger.info(f"Epochs: {args.epochs}")

    best_f1 = 0
    model.to(args.device)

    # ======================= Epoch Loop =======================
    stop_counter = 0

    for epoch in range(args.epochs):

        model.train()
        loss_meter = utils.AverageMeter()

        train_loader = multi_data_loader_st(
            source_inputs, source_labels, target_train_inputs,
            n_batch, args.train_batch_size
        )

        for sinputs, slabels, tinputs in tqdm(train_loader, desc=f"Epoch {epoch+1}", total=n_batch):
            sinputs = [torch.tensor(si).to(args.device) for si in sinputs]
            slabels = [torch.tensor(sl).to(args.device) for sl in slabels]
            tinputs = torch.tensor(tinputs).to(args.device)

            clf_losses, transfer_losses = model(sinputs, tinputs, slabels)
            loss = torch.mean(clf_losses) + args.transfer_loss_weight * torch.mean(transfer_losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item())

        logger.info(f"Epoch {epoch+1}/{args.epochs} | Loss: {loss_meter.avg:.4f}")

        # Evaluate
        result = test(args, model, target_test_inputs, target_test_labels)
        f1 = result["test_f1"]

        if f1 > best_f1:
            best_f1 = f1
            stop_counter = 0

            save_path = os.path.join(args.output_dir, "checkpoint-best-f1-madv", args.model_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(model.state_dict(), save_path)
            logger.info(f"==> BEST MODEL SAVED: f1={round(best_f1,4)}")

        else:
            stop_counter += 1

        if stop_counter >= args.early_stop:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Final Best F1 = {best_f1:.4f}")


# ======================= Testing =======================
def test(args, model, test_inputs, test_labels):

    loader = data_loader(test_inputs, test_labels, args.eval_batch_size, shuffle=False)
    logger.info("***** TEST *****")

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    all_logits, all_true = [], []
    loss_meter = utils.AverageMeter()

    for batch_x, batch_y in loader:
        batch_x = torch.tensor(batch_x).to(args.device)
        batch_y = torch.tensor(batch_y).to(args.device)

        with torch.no_grad():
            logits = model.predict(batch_x)
            loss = criterion(logits, batch_y)

        loss_meter.update(loss.item())
        all_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_true.append(batch_y.cpu().numpy())

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_true)
    y_pred = logits[:, 1] > 0.5

    result = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_recall": float(recall_score(y_true, y_pred)),
        "test_precision": float(precision_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred)),
        "test_auc": float(roc_auc_score(y_true, logits[:, 1])),
        "test_loss": loss_meter.avg
    }

    logger.info("TEST RESULTS:")
    for k, v in result.items():
        logger.info(f"{k}: {round(v,4)}")

    return result


# ======================= Argument Parser =======================
def get_parser():

    parser = argparse.ArgumentParser(description="MSVD Cross-domain Vulnerability Detector")

    # Files
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--source_train_file_list", nargs='+', required=True)
    parser.add_argument("--source_valid_file_list", nargs='+', required=True)
    parser.add_argument("--target_train_file", type=str, required=True)
    parser.add_argument("--target_test_file", type=str, required=True)
    parser.add_argument('--num_class', type=int, default=2, help='number of output classes')
    parser.add_argument("--bottleneck_width",type=int,default=256,help="Dimension of the bottleneck layer")
    parser.add_argument("--output_dir", type=str, required=True)

    # Model
    parser.add_argument("--model_type", type=str, default="roberta")
    parser.add_argument("--model_name", type=str, default="model.bin")
    parser.add_argument("--model_name_or_path", type=str, default="../codebert")

    # FIXED → tokenizer_name added
    parser.add_argument("--tokenizer_name", type=str, default="../codebert",
                        help="Tokenizer name or local path.")

    parser.add_argument("--block_size", type=int, default=512)

    # Train config
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early_stop", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transfer_loss_weight", type=float, default=0.1)

    return parser


# ======================= Model Loader =======================
def get_model(args):

    args.number_domain = len(args.source_train_file_list)

    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=False,
        local_files_only=False
    )

    model = models.MSVD(args)

    return model, tokenizer


# ======================= Main =======================
def main():

    parser = get_parser()
    args = parser.parse_args()

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO
    )

    logger.info(f"Device: {device}, GPUs: {args.n_gpu}")
    set_seed(args)

    # Load model
    model, tokenizer = get_model(args)

    if args.do_train:
        train(model, tokenizer, args)

    if args.do_test:
        x, y = load_dataset(tokenizer, args.target_test_file, args)

        ckpt = os.path.join(args.output_dir, "checkpoint-best-f1-madv", args.model_name)
        model.load_state_dict(torch.load(ckpt, map_location=args.device))

        test(args, model, x, y)


if __name__ == "__main__":
    main()
