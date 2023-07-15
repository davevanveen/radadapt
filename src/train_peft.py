import math
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AdamW,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
)

import constants
import parser
import process


def main():
    # parse arguments, set data and model paths based on expmt params
    args = parser.get_parser()

    # assert valid method
    assert args.case_id >= 100
    assert constants.cases[args.case_id]["method"] in constants.METHODS

    dir_models = os.path.join(args.dir_models_tuned)
    dir_tb_log = os.path.join(dir_models, "logs")
    if not os.path.exists(dir_tb_log):
        os.makedirs(dir_tb_log)

    # init tb writer. via cl: tensorboard --logdir=args.dir_out --port=8888
    writer = SummaryWriter(dir_tb_log)

    # load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(constants.MODELS[args.model])
    model = AutoModelForSeq2SeqLM.from_pretrained(constants.MODELS[args.model])
    config = get_config(constants.cases[args.case_id]["method"])
    model = get_peft_model(model, config).to(args.device)
    model.print_trainable_parameters()

    # preprocess data, create data loaders
    trn_dataset = process.load_data(args, task="trn")
    val_dataset = process.load_data(args, task="val")
    trn_loader = process.get_loader(trn_dataset, tokenizer, args.batch_size)
    val_loader = process.get_loader(val_dataset, tokenizer, args.batch_size)
    args.steps_per_epoch = len(trn_loader)
    print(
        f"{len(trn_dataset)} samples w batch size {args.batch_size}, "
        f"hence {args.steps_per_epoch} gradient steps per epoch"
    )

    # define optimizer, lr scheduler
    num_training_steps = len(trn_loader) * args.trn_epochs
    optimizer, lr_scheduler = define_optimizer(
        model=model, num_training_steps=num_training_steps, args=args
    )

    model.train()
    best_val_loss = math.inf
    patience = (
        constants.PATIENCE
    )  # early stop if loss doesn't reach new min in consec epochs
    n_steps = 0  # track number of steps taken
    trn_losses = []
    print("begin training!")

    for epoch in range(args.trn_epochs):
        with tqdm(total=len(trn_loader)) as pbar:  # progress bar
            for batch in trn_loader:
                n_steps += 1

                # forward pass
                batch.pop("idx")  # indices are irrelevant for training
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)

                # compute loss, gradient step
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()

                # and optimizer step/zero after grad_accum_steps steps
                if (n_steps % args.grad_accum_steps == 0) or (
                    n_steps == len(trn_loader)
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step()

                detached_loss = loss.detach().float()
                trn_losses.append(detached_loss)
                writer.add_scalar("trn_loss", detached_loss, n_steps)
                writer.add_scalar("trn_perplexity", torch.exp(detached_loss), n_steps)
                pbar.update(1)

        # calculate validation loss
        with tqdm(total=len(val_loader)) as pbar:  # progress bar
            val_losses = []
            for batch in val_loader:
                batch.pop("idx")  # indices are irrelevant for training
                batch = {k: v.to(args.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs_val = model(**batch)
                    val_losses.append(outputs_val.loss.detach().float())
                pbar.update(1)

        trn_loss_epoch = sum(trn_losses) / len(trn_losses)
        val_loss_epoch = sum(val_losses) / len(val_losses)
        trn_perplexity_epoch = torch.exp(trn_loss_epoch)
        val_perplexity_epoch = torch.exp(val_loss_epoch)

        writer.add_scalar("lr", lr_scheduler.get_lr()[0], epoch)
        writer.add_scalar("trn_loss_epoch", trn_loss_epoch, epoch)
        writer.add_scalar("val_loss_epoch", val_loss_epoch, epoch)
        writer.add_scalar("trn_perplexity_epoch", trn_perplexity_epoch, epoch)
        writer.add_scalar("val_perplexity_epoch", val_perplexity_epoch, epoch)

        print(
            f"epoch: {epoch}/{args.trn_epochs}, "
            f"trn_loss_epoch: {trn_loss_epoch}, "
            f"val_loss_epoch: {val_loss_epoch}, "
            f"lr: {lr_scheduler.get_lr()[0]}"
        )

        # save model at each epoch
        model_save_dir = os.path.join(dir_models, f"{epoch}")
        model.save_pretrained(model_save_dir)

        # early stopping
        if val_loss_epoch > best_val_loss:
            if patience == 0:
                print(f"Stopping early at epoch {epoch}!")
                break
            else:
                patience -= 1
        else:
            patience = constants.PATIENCE
            best_val_loss = val_loss_epoch


def get_config(method):
    """given a peft method, get model configs
    supported: lora, prefix_tuning"""

    if method == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=constants.LORA_R,
            lora_alpha=constants.LORA_ALPHA,
            lora_dropout=constants.LORA_DROPOUT,
        )

    elif method == "prefix_tuning":
        config = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            num_virtual_tokens=constants.PROMPT_LEN,
        )

    else:
        raise NotImplementedError("peft method not supported")

    return config


def define_optimizer(model, num_training_steps, args):
    """define optimizer given expmt params"""

    # extract learning rate params
    case = constants.cases[args.case_id]
    lr0 = case["lr0"]  # initial learning rate

    # define optimizer, lr_scheduler
    optimizer = AdamW(model.parameters(), lr=lr0, no_deprecation_warning=True)

    if case["lr_schedule"] == "polynomial_decay":
        lrn = case["lrn"]  # final learning rate
        lr_decay_power = case["lr_decay_power"]  # rate of polynomial decay
        str_ = f"Using polynomial decay scheduler with lr0 {lr0}, "
        str_ += f"lrn {lrn}, power {lr_decay_power},"

        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            lr_end=lrn,
            power=lr_decay_power,
            num_warmup_steps=args.lr_n_warmup_steps,
            num_training_steps=num_training_steps,
        )

    elif case["lr_schedule"] == "linear_decay":
        str_ = f"Using linear scheduler with lr0 {lr0},"
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_n_warmup_steps,
            num_training_steps=num_training_steps,
        )

    elif case["lr_schedule"] == "constant":
        str_ = f"Using constant learning rate {lr0},"
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.lr_n_warmup_steps,
        )

    else:
        raise NotImplementedError("learning rate method not supported")

    str_ += f" and {args.lr_n_warmup_steps} warm-up steps!"
    print(str_)

    return optimizer, lr_scheduler


if __name__ == "__main__":
    main()
