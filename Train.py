"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
Edited by:
    Gal Suchetzky (galsuchetzky@gmail.com)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import Utils

from Args import get_train_args
from collections import OrderedDict
from json import dumps
# from models import <your model> TODO: import your model.
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load

# TODO move to utils
def start_logger_and_devices(args):
    # Set up logging and devices
    args.save_dir = Utils.get_save_dir(args.save_dir, args.name, training=True)
    log = Utils.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = Utils.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))
    return log, device

# TODO move to utils
def set_seed(args, log):
    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main(args):
    # Setup
    log, device = start_logger_and_devices(args)
    set_seed(args, log)
    # TODO: add setup code, for example load word vectors, etc...

    # # Get embeddings
    # log.info('Loading embeddings...')
    # word_vectors = util.torch_from_json(args.word_emb_file)

    # Initialize model
    # TODO: edit this code to initialize your model
    log.info('Building model...')
    # model = None  # TODO: edit this.
    # model = nn.DataParallel(model, args.gpu_ids)

    if args.load_path:
        log.info(f'Loading checkpoint from {args.load_path}...')
        model, step = Utils.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0

    model = model.to(device)
    model.train()
    ema = Utils.EMA(model, args.ema_decay)

    # TODO get everything from config, not args
    # Get saver
    saver = Utils.CheckpointSaver(args.save_dir,
                                  max_checkpoints=args.max_checkpoints,
                                  metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric,
                                  log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)
    scheduler = sched.LambdaLR(optimizer, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')

    train_dataset = None  # TODO: initialize your train dataset with the utils Dataset class.
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers)
    dev_dataset = None  # TODO: initialize your dev dataset with the utils Dataset class.
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

    # Train
    log.info('Training...')
    # Start training
    trainer = Trainer(model, config, helper, logger)
    logger.info("Starting training...", )
    trainer.train(train_examples, dev_examples)

    # steps_till_eval = args.eval_steps
    # epoch = step // len(train_dataset)
    # while epoch != args.num_epochs:
    #     epoch += 1
    #     log.info(f'Starting epoch {epoch}...')
    #     with torch.enable_grad(), \
    #          tqdm(total=len(train_loader.dataset)) as progress_bar:
    #         # TODO edit the for loop for your needs.
    #         for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
    #             # Setup for forward
    #             # TODO: send the data to device and zero the grad.
    #             # cw_idxs = cw_idxs.to(device)
    #             # qw_idxs = qw_idxs.to(device)
    #             batch_size = 0  # cw_idxs.size(0)
    #             # optimizer.zero_grad()
    #
    #             # Forward
    #             # TODO: Run a forward pass.
    #             # log_p1, log_p2 = model(cw_idxs, qw_idxs)
    #             # y1, y2 = y1.to(device), y2.to(device)
    #             loss = 0  # F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
    #             loss_val = 0  # loss.item()
    #
    #             # Backward
    #             # TODO: Run a backward pass.
    #             # loss.backward()
    #             # nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #             # optimizer.step()
    #             # scheduler.step(step // batch_size)
    #             # ema(model, step // batch_size)
    #
    #             # Log info
    #             step += batch_size
    #             progress_bar.update(batch_size)
    #             progress_bar.set_postfix(epoch=epoch,
    #                                      NLL=loss_val)
    #             tbx.add_scalar('train/NLL', loss_val, step)
    #             tbx.add_scalar('train/LR',
    #                            optimizer.param_groups[0]['lr'],
    #                            step)
    #
    #             steps_till_eval -= batch_size
    #             if steps_till_eval <= 0:
    #                 steps_till_eval = args.eval_steps
    #
    #                 # Evaluate and save checkpoint
    #                 log.info(f'Evaluating at step {step}...')
    #                 ema.assign(model)
    #                 results, pred_dict = evaluate(model, dev_loader, device,
    #                                               args.dev_eval_file,
    #                                               args.max_ans_len,
    #                                               args.use_squad_v2)
    #                 saver.save(step, model, results[args.metric_name], device)
    #                 ema.resume(model)
    #
    #                 # Log to console
    #                 results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    #                 log.info(f'Dev {results_str}')
    #
    #                 # Log to TensorBoard
    #                 log.info('Visualizing in TensorBoard...')
    #                 for k, v in results.items():
    #                     tbx.add_scalar(f'dev/{k}', v, step)
    #                 Utils.visualize(tbx,
    #                                 pred_dict=pred_dict,
    #                                 eval_path=args.dev_eval_file,
    #                                 step=step,
    #                                 split='dev',
    #                                 num_visuals=args.num_visuals)

    # Save predictions of the best model
    logger.info("Training completed, saving predictions of the best model...", )
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_output))
        model.eval()
        predictor = Predictor(model, config)
        output = predictor.predict(dev_examples, use_str_labels=True)
        sentences, labels, predictions = zip(*output)
        predictions = [[LBLS[l] for l in preds] for preds in predictions]
        output = list(zip(sentences, labels, predictions))

        with open(model.config.conll_output, 'w') as f:
            write_conll(f, output)
        with open(model.config.eval_output, 'w') as f:
            for sentence, labels, predictions in output:
                print_sentence(f, sentence, labels, predictions)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
    # TODO: edit to evaluate your model.
    nll_meter = Utils.AverageMeter()

    model.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
         tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1, log_p2 = model(cw_idxs, qw_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = Utils.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = Utils.convert_tokens(gold_dict,
                                            ids.tolist(),
                                            starts.tolist(),
                                            ends.tolist(),
                                            use_squad_v2)
            pred_dict.update(preds)

    model.train()

    results = Utils.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
