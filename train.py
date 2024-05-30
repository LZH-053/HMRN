import os
import time
import shutil

import torch
import numpy as np
import pickle, random

import data
import opts
from model import HMRN
from evaluation_individual import eval_val, AverageMeter, LogCollector, encode_data, shard_attn_scores_val

import logging
# import tensorboard_logger as tb_logger
from vg import vg
from vocab import Vocabulary, deserialize_vocab

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    opt = opts.parse_opt()
    print(opt)
    # LOG
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # tb_logger.configure(opt.logger_name, flush_secs=5)
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 训练开始的时间

    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    # Load dataset
    train_db = vg(opt, 'train')
    val_db   = vg(opt, 'val')

    # Load data loaders
    train_loader, val_loader = data.get_loaders(train_db, val_db, opt.batch_size, opt.workers, opt.pin_memory)

    # Construct the model
    model = HMRN(opt)

    # load model and options
    if opt.model_path is not None:
        checkpoint = torch.load(opt.model_path)
        # opt = checkpoint['opt']
        save_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        print(opt)
        # load model state
        model.load_state_dict(checkpoint['model'])

    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        # learning rate decay
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        r_sum = validate(opt, val_loader, model, epoch)

        # remember best R@ sum and save checkpoint
        is_best = r_sum > best_rsum
        best_rsum = max(r_sum, best_rsum)

        # save checkpoint
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/', time=t)


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                .format(
                    epoch, i, len(train_loader), e_log=str(model.logger)))

    # validate at every epoch
        # if model.Eiters % opt.val_step == 0:
        #     validate(opt, val_loader, model, epoch)


def validate(opt, val_loader, model, epoch):
    # compute the encoding for all the validation images and captions
    with torch.no_grad(): 
        img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)     

        # record computation time of validation
        start = time.time()
        sims = shard_attn_scores_val(model, img_embs, cap_embs, cap_lens, opt, shared_size=250)
        end = time.time()
        print("calculate similarity time:", end-start)

        # image retrieval
        (r1i, r5i, r10i, meanr) = eval_val(img_embs, cap_embs, cap_lens, sims)
        logging.info("10th round image retrieval results: R1: %.1f, R5: %.1f, R10: %.1f, MR: %.1f" % (r1i, r5i, r10i, meanr))

        # sum of recalls to be used for early stopping
        r_sum = r1i + r5i + r10i                                                                         
    
    torch.cuda.empty_cache()

    return r_sum


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', time=''):
    if is_best:
        torch.save(state, prefix + time + ' model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    multiplies 0.1 after every 75 epoch
    """
    if epoch > 0 and epoch % opt.lr_update == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


if __name__ == '__main__':
    main()
