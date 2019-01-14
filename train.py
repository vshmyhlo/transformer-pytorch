from tensorboardX import SummaryWriter
import torch.nn.functional as F
from ticpfptp.metrics import Mean
from ticpfptp.torch import load_weights, save_model, fix_seed
from ticpfptp.format import args_to_path, args_to_string
from tqdm import tqdm
import logging
import torch.utils.data
import os
import argparse
import torch
import torch.optim as optim
import transformer
from dataset import TrainEvalDataset
from nltk.translate.bleu_score import sentence_bleu


# TODO: rename y, y_top, etc.
# TODO: check dropout
# TODO: scheduling
# TODO: wer
# TODO: remove buckets and simplify code
# TODO: try lowercase everything
# TODO: visualize attention
# TODO: beam search
# TODO: add requirements.txt file
# TODO: byte pair encoding
# TODO: compute bleu (https://machinelearningmastery.com/calculate-bleu-score-for-text-python/)
# TODO: add option to share embedding and projection weights
# TODO: embedding and projection weights scaling
# TODO: learning_rate scheduling
# TODO: weight initialization
# TODO: try disable share_embedding
# TODO: test masking
# TODO: label smoothing


def compute_loss(logits, y):
    # TODO: use ignore_index argument
    # TODO: sum by time and mean by batch

    non_padding = y != 0
    loss = F.cross_entropy(logits[non_padding], y[non_padding])

    return loss


# TODO: debug collate fn
# TODO: revisit
# TODO: return lengths

def pad_and_pack(seqs):
    max_len = max(len(seq) for seq in seqs)
    seqs = [seq + [0] * (max_len - len(seq)) for seq in seqs]
    seqs = torch.tensor(seqs)

    return seqs


def collate_fn(samples):
    x, y = zip(*samples)

    x = pad_and_pack(x)
    y = pad_and_pack(y)

    return x, y


def build_parser():
    # TODO: revisit
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-path", type=str, default='./tf_log')
    parser.add_argument('--restore-path', type=str)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--share-embedding", action='store_true')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset-path", type=str, nargs=3, default=['./iwslt15', 'en', 'vi'])
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-threads", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, choices=['adam', 'momentum'], default='adam')
    parser.add_argument(
        "--attention-type", type=str, choices=['scaled_dot_product', 'luong'], default='scaled_dot_product')

    return parser


def take_until_token(seq, token):
    if token in seq:
        return seq[:seq.index(token)]
    else:
        return seq


def compute_bleu(logits, y, eos_id):
    true = y.data.cpu().numpy()
    pred = logits.argmax(-1).data.cpu().numpy()

    bleus = []
    for t, p in zip(true, pred):
        t = take_until_token(list(t), eos_id)
        p = take_until_token(list(p), eos_id)
        bleus.append(sentence_bleu(references=[t], hypothesis=p))

    return bleus


# TODO: try larger betas
# TODO: revisit
def build_optimizer(parameters, optimizer, learning_rate):
    if optimizer == 'adam':
        return optim.Adam(
            parameters, lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer == 'momentum':
        return optim.SGD(parameters, lr=learning_rate, momentum=0.9)


def main():
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    logging.info(args_to_string(args))
    experiment_path = os.path.join(
        args.experiment_path,
        args_to_path(args, ignore=['experiment_path', 'restore_path', 'seed', 'epochs', 'n_threads', 'dataset_path']))
    fix_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = TrainEvalDataset(
        args.dataset_path[0], subset='train', source=args.dataset_path[1], target=args.dataset_path[2])
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_threads,
        collate_fn=collate_fn)
    eval_dataset = TrainEvalDataset(
        args.dataset_path[0], subset='tst2012', source=args.dataset_path[1], target=args.dataset_path[2])
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        collate_fn=collate_fn)

    model = transformer.Tranformer(
        source_vocab_size=len(train_dataset.source_vocab),
        target_vocab_size=len(train_dataset.target_vocab),
        size=args.size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        attention_type=args.attention_type,
        share_embedding=args.share_embedding)
    model.to(device)
    if args.restore_path is not None:
        load_weights(model, os.path.join(args.restore_path))

    optimizer = build_optimizer(
        model.parameters(),
        args.optimizer,
        learning_rate=args.learning_rate)

    train_writer = SummaryWriter(experiment_path)
    eval_writer = SummaryWriter(os.path.join(experiment_path, 'eval'))
    metrics = {'loss': Mean(), 'bleu': Mean()}

    for epoch in range(args.epochs):
        # train
        model.train()
        for x, y in tqdm(train_data_loader, desc='epoch {} training'.format(epoch)):
            x, y = x.to(device), y.to(device)
            y_bottom, y = y[:, :-1], y[:, 1:]

            logits = model(x, y_bottom)
            loss = compute_loss(logits=logits, y=y)
            metrics['loss'].update(loss.data.cpu().numpy())

            optimizer.zero_grad()
            loss.mean().backward()  # TODO: sum/mean non padding
            optimizer.step()
            break

        train_writer.add_scalar('loss', metrics['loss'].compute_and_reset(), global_step=epoch)

        # eval
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(eval_data_loader, desc='epoch {} evaluating'.format(epoch)):
                x, y = x.to(device), y.to(device)
                y_bottom, y = y[:, :-1], y[:, 1:]

                logits = model(x, y_bottom)
                loss = compute_loss(logits=logits, y=y)
                metrics['loss'].update(loss.data.cpu().numpy())

                bleu = compute_bleu(logits=logits, y=y, eos_id=train_dataset.target_vocab.eos_id)
                metrics['bleu'].update(bleu)

        eval_writer.add_scalar('loss', metrics['loss'].compute_and_reset(), global_step=epoch)
        eval_writer.add_scalar('bleu', metrics['bleu'].compute_and_reset(), global_step=epoch)

        # save model
        save_model(model, experiment_path)


if __name__ == '__main__':
    main()
