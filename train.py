import os
import sys
import random
import shutil
import logging
import argparse
import subprocess
from time import time

import numpy as np
import torch

from test import test
from lib.config import Config
from utils.evaluator import Evaluator

# Hàm lưu trạng thái huấn luyện (model, optimizer, lr_scheduler)
def save_train_state(path, model, optimizer, lr_scheduler, epoch):
    train_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(train_state, path)
    logging.info(f"Checkpoint saved at {path}")



# Hàm load checkpoint để tiếp tục huấn luyện
# Hàm load checkpoint để tiếp tục huấn luyện
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # checkpoint = torch.load(checkpoint_path)  # Thay 'torch.load_state_dict' bằng 'torch.load'
    # print(type(checkpoint))  # In ra kiểu dữ liệu của checkpoint
    # print(checkpoint)  # In ra nội dung của checkpoint để kiểm tra cấu trúc

    # Kiểm tra cấu trúc checkpoint
    if isinstance(checkpoint, dict):
        # load model với GPU
        model.load_state_dict(checkpoint['model'])
        # lode model với CPU
        # model.load_state_dict(torch.load(os.path.join(exp_root, "models", "model_{:03d}.pt".format(epoch)), map_location='cpu')['model']) 
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        epoch = checkpoint['epoch']
    else:
        raise ValueError(f"Checkpoint at {checkpoint_path} không có cấu trúc hợp lệ!")

    logging.info(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {epoch}")
    return model, optimizer, scheduler, epoch




# Hàm huấn luyện model
# Hàm huấn luyện model
def train(model, train_loader, exp_dir, cfg, val_loader, train_state=None):
    optimizer = cfg.get_optimizer(model.parameters())
    scheduler = cfg.get_lr_scheduler(optimizer)
    starting_epoch = 1

    # Nếu train_state không phải là None, load trạng thái huấn luyện từ checkpoint
    if train_state is not None:
        model = train_state[0]  # load model từ checkpoint
        optimizer = train_state[1]  # load optimizer từ checkpoint
        scheduler = train_state[2]  # load scheduler từ checkpoint
        starting_epoch = train_state[3] + 1  # Epoch sẽ bắt đầu từ tiếp theo
        scheduler.step(starting_epoch)

    # Cấu hình và tham số huấn luyện
    criterion_parameters = cfg.get_loss_parameters()
    criterion = model.loss
    total_step = len(train_loader)
    ITER_LOG_INTERVAL = cfg['iter_log_interval']
    ITER_TIME_WINDOW = cfg['iter_time_window']
    MODEL_SAVE_INTERVAL = cfg['model_save_interval']
    CHECKPOINT_SAVE_INTERVAL = cfg['checkpoint_save_interval']
    t0 = time()
    total_iter = 0
    iter_times = []
    logging.info("Starting training.")
    
    for epoch in range(starting_epoch, num_epochs + 1):
        epoch_t0 = time()
        logging.info(f"Beginning epoch {epoch}")
        accum_loss = 0
        for i, (images, labels, img_idxs) in enumerate(train_loader):
            total_iter += 1
            iter_t0 = time()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, epoch=epoch)
            loss, loss_dict_i = criterion(outputs, labels, **criterion_parameters)
            accum_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_times.append(time() - iter_t0)
            if len(iter_times) > 100:
                iter_times = iter_times[-ITER_TIME_WINDOW:]
            if (i + 1) % ITER_LOG_INTERVAL == 0:
                loss_str = ', '.join(
                    ['{}: {:.4f}'.format(loss_name, loss_dict_i[loss_name]) for loss_name in loss_dict_i])
                logging.info(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {accum_loss / (i+1):.4f} ({loss_str}), s/iter: {np.mean(iter_times):.4f}, lr: {optimizer.param_groups[0]['lr']:.1e}")

            # Lưu checkpoint
            if total_iter % CHECKPOINT_SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(exp_dir, "models", f"checkpoint_{total_iter:07d}.pt")
                save_train_state(checkpoint_path, model, optimizer, scheduler, epoch)

        logging.info(f"Epoch time: {time() - epoch_t0:.4f}")

        # Lưu model sau mỗi epoch
        if epoch % MODEL_SAVE_INTERVAL == 0 or epoch == num_epochs:
            model_path = os.path.join(exp_dir, "models", f"model_{epoch:03d}.pt")
            save_train_state(model_path, model, optimizer, scheduler, epoch)

        if val_loader is not None:
            evaluator = Evaluator(val_loader.dataset, exp_root)
            evaluator, val_loss = test(
                model,
                val_loader,
                evaluator,
                None,
                cfg,
                view=False,
                epoch=-1,
                verbose=False,
            )
            _, results = evaluator.eval(label=None, only_metrics=True)
            logging.info(f"Epoch [{epoch}/{num_epochs}], Val loss: {val_loss:.4f}")
            model.train()

        scheduler.step()

    logging.info(f"Training time: {time() - t0:.4f}")

    return model




def parse_args():
    parser = argparse.ArgumentParser(description="Train PolyLaneNet")
    parser.add_argument("--exp_name", default="default", help="Experiment name", required=True)
    parser.add_argument("--cfg", default="config.yaml", help="Config file", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--validate", action="store_true", help="Validate model during training")
    parser.add_argument("--deterministic",
                        action="store_true",
                        help="set cudnn.deterministic = True and cudnn.benchmark = False")

    return parser.parse_args()


def get_code_state():
    state = "Git hash: {}".format(
        subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8'))
    state += '\n*************\nGit diff:\n*************\n'
    state += subprocess.run(['git', 'diff'], stdout=subprocess.PIPE).stdout.decode('utf-8')

    return state


def setup_exp_dir(exps_dir, exp_name, cfg_path):
    dirs = ["models"]
    exp_root = os.path.join(exps_dir, exp_name)

    for dirname in dirs:
        os.makedirs(os.path.join(exp_root, dirname), exist_ok=True)

    shutil.copyfile(cfg_path, os.path.join(exp_root, 'config.yaml'))
    with open(os.path.join(exp_root, 'code_state.txt'), 'w') as file:
        file.write(get_code_state())

    return exp_root


# Hàm lấy trạng thái huấn luyện từ checkpoint
def get_exp_train_state(exp_root):
    models_dir = os.path.join(exp_root, "models")
    models = [name for name in os.listdir(models_dir) if name.endswith(".pt")]
    
    if not models:
        logging.warning("No checkpoint found!")
        return None

    # Sắp xếp và chọn checkpoint có epoch cao nhất
    last_epoch, last_modelname = sorted(
        [(int(name.split("_")[1].split(".")[0]), name) for name in models],
        key=lambda x: x[0],
    )[-1]
    
    checkpoint_path = os.path.join(models_dir, last_modelname)
    model = cfg.get_model().to(device)  # Khởi tạo model trước khi load
    optimizer = cfg.get_optimizer(model.parameters())  # Khởi tạo optimizer
    scheduler = cfg.get_lr_scheduler(optimizer)  # Khởi tạo scheduler

    # Load checkpoint vào model, optimizer, scheduler
    model, optimizer, scheduler, epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    
    logging.info(f"Loaded train state from {checkpoint_path} (epoch {epoch})")
    
    return model, optimizer, scheduler, epoch




def log_on_exception(exc_type, exc_value, exc_traceback):
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args.cfg)

    # Set up seeds
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set up experiment
    if not args.resume:
        exp_root = setup_exp_dir(cfg['exps_dir'], args.exp_name, args.cfg)
    else:
        exp_root = os.path.join(cfg['exps_dir'], os.path.basename(os.path.normpath(args.exp_name)))

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "log.txt")),
            logging.StreamHandler(),
        ],
    )

    sys.excepthook = log_on_exception

    logging.info("Experiment name: {}".format(args.exp_name))
    logging.info("Config:\n" + str(cfg))
    logging.info("Args:\n" + str(args))

    # Get data sets
    train_dataset = cfg.get_dataset("train")

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    num_epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]

    # Model
    model = cfg.get_model().to(device)

    train_state = None
    if args.resume:
        train_state = get_exp_train_state(exp_root)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    if args.validate:
        val_dataset = cfg.get_dataset("val")
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=8)
    # Train regressor
    try:
        model = train(
            model,
            train_loader,
            exp_root,
            cfg,
            val_loader=val_loader if args.validate else None,
            train_state=train_state,
        )
    except KeyboardInterrupt:
        logging.info("Training session terminated.")
    test_epoch = -1
    if cfg['backup'] is not None:
        subprocess.run(['rclone', 'copy', exp_root, '{}/{}'.format(cfg['backup'], args.exp_name)])

    # Eval model after training
    # Eval model after training
    test_dataset = cfg.get_dataset("test")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)

    evaluator = Evaluator(test_loader.dataset, exp_root)

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(exp_root, "test_log.txt")),
            logging.StreamHandler(),
        ],
    )
    logging.info('Code state:\n {}'.format(get_code_state()))
    
    # Tiến hành đánh giá mô hình trên dữ liệu test
    _, mean_loss = test(model, test_loader, evaluator, exp_root, cfg, epoch=test_epoch, view=False)
    logging.info("Mean test loss: {:.4f}".format(mean_loss))

    evaluator.exp_name = args.exp_name  # Lưu tên experiment

    # Đánh giá mô hình và ghi kết quả
    eval_str, _ = evaluator.eval(label='{}_{}'.format(os.path.basename(args.exp_name), test_epoch))

    logging.info(eval_str)  # In kết quả đánh giá ra log

