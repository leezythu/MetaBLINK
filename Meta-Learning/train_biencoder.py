# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import math
import argparse
import pickle
import torch
import json
import sys
import io
import random
import time
import numpy as np

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder
import logging

import blink.candidate_ranking.utils as utils
import blink.biencoder.data_process as data
from blink.biencoder.zeshel_utils import DOC_PATH, WORLDS, world_to_id
from blink.common.optimizer import get_bert_optimizer
from blink.common.params import BlinkParser
from blink.magic import MagicModule
import torch.nn.functional as F

from blink.biencoder.eval_biencoder import encode_candidate
from blink.biencoder.eval_biencoder import nnquery
import higher
logger = None
class MetaDataManager(object):
    def __init__(self):
        super().__init__()
    def set_data_loader(self, meta_loader):
        self.meta_loader = meta_loader
    def reset_meta_iter(self):
        self.meta_iterator = iter(self.meta_loader)
    def generate_meta_inputs(self):
        try:
            meta_batch = next(self.meta_iterator)
        except StopIteration:
            self.reset_meta_iter()
            meta_batch = next(self.meta_iterator)
        return meta_batch

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(
    reranker, eval_dataloader, params, device, logger,
):
    # reranker.model.eval()
    # if params["silent"]:
    #     iter_ = eval_dataloader
    # else:
    #     iter_ = tqdm(eval_dataloader, desc="Evaluation")

    # results = {}

    # eval_accuracy = 0.0
    # nb_eval_examples = 0
    # nb_eval_steps = 0
    # eval_tot_loss = 0
    # for step, batch in enumerate(iter_):
    #     batch = tuple(t.to(device) for t in batch)
    #     context_input, candidate_input, _, _ = batch
    #     with torch.no_grad():
    #         eval_loss, logits = reranker(context_input, candidate_input)
    #     eval_tot_loss += eval_loss
    #     logits = logits.detach().cpu().numpy()
    #     # Using in-batch negatives, the label ids are diagonal
    #     label_ids = torch.LongTensor(
    #             # torch.arange(params["eval_batch_size"])
    #             torch.arange(logits.shape[0])
    #     ).numpy()
    #     tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

    #     eval_accuracy += tmp_eval_accuracy

    #     nb_eval_examples += context_input.size(0)
    #     nb_eval_steps += 1

    # normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    # logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    # avg_loss  = eval_tot_loss.item() / nb_eval_steps
    # logger.info("Eval loss: %.5f" % avg_loss )
    # results["normalized_accuracy"] = normalized_eval_accuracy
    # return results
    cand_pool_path = "test_cand_all"
    candidate_pool = torch.load(cand_pool_path)
    candidate_encoding = encode_candidate(
        reranker,
        candidate_pool,
        params["encode_batch_size"],
        silent=params["silent"],
        logger=logger,
        is_zeshel=True
    )
    _,res = nnquery.get_topk_predictions(
        reranker,
        eval_dataloader,
        candidate_pool,
        candidate_encoding,
        params["silent"],
        logger,
        params["top_k"],
        True,
        False,
    )
    r64 = res.hits[5] / float(res.cnt)
    print("r64:",r64)
    return r64

def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    # for n,p in reranker.named_parameters():
    #     print(n)
    # exit(0)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    # meta_batch_size = params["meta_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    mode = params["train_mode"]
    # mode = "test"
    # Load train data 
    if not os.path.exists(mode+"_tensor_data"):#only for zeshel
        train_samples = utils.read_dataset(mode, params["data_path"])
        logger.info("Read %d train samples." % len(train_samples))

        train_data, train_tensor_data = data.process_mention_data(
            train_samples,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            context_key=params["context_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            negative = False
        )
        print("saving "+mode+"_tensor_data....")
        torch.save(train_tensor_data,mode+"_tensor_data")
    print("loading "+mode+" tensor data...")
    train_tensor_data = torch.load(mode+"_tensor_data")
    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
    )
    # Load meta data
    mode = params["meta_mode"]
    meta_samples = utils.read_dataset(mode, params["data_path"])
    logger.info("Read %d meta samples." % len(meta_samples))

    meta_data, meta_tensor_data = data.process_mention_data(
        meta_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        negative = False
    )
    meta_sampler = RandomSampler(meta_tensor_data)
    meta_dataloader = DataLoader(
        meta_tensor_data, sampler=meta_sampler, batch_size=train_batch_size
    )
    meta_manager = MetaDataManager()
    meta_manager.set_data_loader(meta_dataloader)
    meta_manager.reset_meta_iter()

    # Load eval data
    # TODO: reduce duplicated code here
    mode = params["valid_mode"]
    if not os.path.exists(mode+"_tensor_data"):
        valid_samples = utils.read_dataset(mode, params["data_path"])
        logger.info("Read %d valid samples." % len(valid_samples))

        valid_data, valid_tensor_data = data.process_mention_data(
            valid_samples,
            tokenizer,
            params["max_context_length"],
            params["max_cand_length"],
            context_key=params["context_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            negative = False
        )
        print("saving "+mode+"_tensor_data....")
        torch.save(valid_tensor_data,mode+"_tensor_data")
    print("loading "+mode+" tensor data...")
    valid_tensor_data = torch.load(mode+"_tensor_data")
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )

    # # evaluate before training
    # results = evaluate(
    #     reranker, valid_dataloader, params, device=device, logger=logger,
    # )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    # optimizer = get_optimizer(model, params)
    optimizer = get_optimizer(reranker, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)
    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")
        optimizer.zero_grad()
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')
        for step, batch in enumerate(iter_):
            #noisy batch
            batch = tuple(t.to(cuda0) for t in batch)
            context_input, candidate_input, _, _ = batch
            with higher.innerloop_ctx(reranker, optimizer,device=cuda1) as (meta_model, meta_opt):
                meta_model.device = cuda1
                context_input, candidate_input = context_input.to(cuda1), candidate_input.to(cuda1)
                # 1. Update meta model on training data
                _, scores = meta_model(context_input, candidate_input)
                target = torch.LongTensor(torch.arange(scores.shape[0]))
                target = target.to(cuda1)
                meta_train_loss = F.cross_entropy(scores, target, reduction="none")
                # print("meta_train_loss",meta_train_loss)
                eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device=cuda1)
                meta_train_loss = torch.sum(eps * meta_train_loss)
                # print("opt.device",meta_opt.device)
                meta_opt.step(meta_train_loss)
                # if n_gpu > 1:
                #     loss = loss.mean() # mean() to average on multi-gpu.
                # 2. Compute grads of eps on meta validation data
                meta_batch = meta_manager.generate_meta_inputs()
                meta_batch = tuple(t.to(cuda1) for t in meta_batch)
                context_meta, candidate_meta, _, _ = meta_batch
                meta_val_loss, _ = meta_model(context_meta, candidate_meta)
                # print("meta_val_loss",meta_val_loss)
                eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()
            # 3. Compute weights for current training batch
            print("eps_grads:",eps_grads)
            w_tilde = torch.clamp(-eps_grads, min=0)
            l1_norm = torch.sum(w_tilde)
            if l1_norm != 0:
                w = w_tilde / l1_norm
            else:
                w = w_tilde

            w = w.to(cuda0)
            print("w:",w)
            context_input = context_input.to(cuda0)
            candidate_input = candidate_input.to(cuda0)
            _, scores = reranker(context_input, candidate_input)
            target = torch.LongTensor(torch.arange(scores.shape[0]))
            target = target.to(cuda0)
            loss = F.cross_entropy(scores, target, reduction="none")
            loss = torch.sum(w * loss)
            # loss, _ = reranker(context_input, candidate_input)
            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward() 
            # report_memory("idx {}".format(0))
            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate(
                    reranker, valid_dataloader, params, device=device, logger=logger,
                )
                # logger.info("***** Saving fine - tuned model *****")
                # epoch_output_folder_path = os.path.join(
                #     model_output_path, "epoch_{}_step_{}".format(epoch_idx,step)
                # )
                # utils.save_model(model, tokenizer, epoch_output_folder_path)
                # exit(0)
                model.train()
                logger.info("\n")
        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(
            model_output_path, "epoch_{}".format(epoch_idx)
        )
        utils.save_model(model, tokenizer, epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker, valid_dataloader, params, device=device, logger=logger,
        )

        # ls = [best_score, results["normalized_accuracy"]]
        ls = [best_score, results]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, 
        "epoch_{}".format(best_epoch_idx),
        WEIGHTS_NAME,
    )
    reranker = load_biencoder(params)
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, logger=logger)


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()
    parser.add_argument(
        "--no_title",
        action="store_true",
    )
    parser.add_argument(
        "--train_mode",
        default="pseudo_sample_yugioh",
        type=str,
    )
    parser.add_argument(
        "--valid_mode",
        default="valid_yugioh",
        type=str,
    )
    parser.add_argument(
        "--meta_mode",
        default="meta_yugioh",
        type=str,
    )
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
