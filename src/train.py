import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

import process_data
import labels
import utils
from config import Config
from logger import setup_logger, get_logger

from sklearn.model_selection import train_test_split

from dataloader import CallUtterance_Dataset, CallTranscript_Dataset
from models import BertModel_Utt, HiBert
import models.metrics as module_metric
from trainer import Trainer, CallTrainer
from asymmetricloss import AsymmetricLoss 

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--config_dir', default='experiments/base', help="Directory containing params.json")


def main():

    args = parser.parse_args()
    print(args.data_dir, args.config_dir)
    config_file = Path(args.config_dir)/'params.json'
    config = Config(config_file)
    print(config.data)

    utils.seed_everything(config.seed)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_logger(Path('.'))
    logger = get_logger('train')

    utt_calldata_df = process_data.read_utterance(Path(args.data_dir)/config.data['utt_corpus'])
    utt_calldata_df = utt_calldata_df.iloc[:200]
    print(utt_calldata_df.shape)
    logger.info(f'utterance data shape: {utt_calldata_df.shape}')

    label2id = {}
    for i, cls in labels.id2label.items():
        label2id[cls] = i

    utt_calldata_df['call_reason'] = utt_calldata_df['call_label'].replace(label2id)
 
    utt_train_df, utt_val_df = train_test_split(
                                    utt_calldata_df,
                                    #stratify=utt_calldata_df['call_reason'],
                                    test_size=0.2, random_state=config.seed)

    tokenizer = BertTokenizer.from_pretrained(config.utt_model_config['model_arch'])
    bertconfig = BertConfig.from_pretrained(
            config.utt_model_config['model_arch'],
            num_labels=len(label2id),
            finetuning_task="text_classification",
            label2id=label2id,
            id2label=labels.id2label,
        )

    config.num_labels = len(label2id)
    device = torch.device(config.device)
    bertmodel = BertModel_Utt.from_pretrained(config.utt_model_config['model_arch'], 
                                              config=bertconfig)
    bertmodel = bertmodel.to(device)

    train_dataset = CallUtterance_Dataset(tokenizer, utt_train_df,
                                          config.utt_model_config)
    val_dataset = CallUtterance_Dataset(tokenizer, utt_val_df,
                                        config.utt_model_config)

    train_loader = DataLoader(train_dataset, batch_size=config.utt_batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.utt_batch_size,
                            shuffle=False, drop_last=True)

    EPOCHS = config.num_epochs
    num_train_steps = len(train_loader) * EPOCHS
    opt = AdamW(bertmodel.parameters(), lr=config.learning_rate, eps=1e-8)
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    marked_criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    teacher_training = nn.KLDivLoss(reduction='sum')
    scheduler = get_linear_schedule_with_warmup(opt, 
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)
    bertmodel = bertmodel.to(device)

    metric_fns = [getattr(module_metric, metric) for metric in config.metrics]
    trainer = Trainer(bertmodel, opt, criterion, marked_criterion, 
                      config, logger, train_loader, device,
                      metric_fns=metric_fns, scheduler=scheduler,
                      val_data_loader=val_loader)
    trainer.train()

    calldata_df = process_data.read_call_transcript(
                    Path(args.data_dir)/config.data['call_corpus'])
    logger.info(f'call data shape: {calldata_df.shape}')

    calldata_df = calldata_df.iloc[:40]
    train_df, val_df = train_test_split(calldata_df, test_size=0.2,
                                        random_state=config.seed)

    train_dataset = CallTranscript_Dataset(tokenizer, train_df,
                                           config.hibert_model_config,
                                           config.num_labels, label2id)
    
    val_dataset = CallTranscript_Dataset(tokenizer, val_df,
                                         config.hibert_model_config,
                                         config.num_labels, label2id)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True)

    hibert_model = HiBert(bertmodel, bertconfig, config.hibert_model_config,
                          label2id, device)
    hibert_model = hibert_model.to(device)

    metric_fns = [getattr(module_metric, metric) for metric in config.call_metrics]
    calltrainer = CallTrainer(hibert_model, opt, criterion, marked_criterion,
                              teacher_training, config, logger, train_loader, device,
                              metric_fns=metric_fns, scheduler=scheduler,
                              val_data_loader=val_loader)
    calltrainer.train()


if __name__ == '__main__':
    main()