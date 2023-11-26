import torch
import random
import nltk

import logging
logger = logging.getLogger(__name__)


def process_caption_bigru(vocab, caption, drop):
    # Convert caption (string) to word ids.
    tokens = ['<start>', ]
    tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
    tokens.append('<end>')

    deleted_idx = []
    for i, token in enumerate(tokens):
        if token in ['<start>', '<end>']:
            tokens[i] = vocab(token)
            continue
        prob = random.random()

        if prob < 0.20 and drop:
            prob /= 0.20
            # 50% randomly change token to mask token
            if prob < 0.5:
                tokens[i] = vocab.word2idx['<mask>']
            # 10% randomly change token to random token
            elif prob < 0.6:
                tokens[i] = random.randrange(len(vocab))
            # 40% randomly remove the token
            else:
                tokens[i] = vocab(token)
                deleted_idx.append(i)
        else:
            tokens[i] = vocab(token)

    if len(deleted_idx) != 0:
        tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]

    return torch.Tensor(tokens)


def process_caption_bert(tokenizer, caption, drop):
    tokens = []
    deleted_idx = []

    basic_tokens = tokenizer.basic_tokenizer.tokenize(caption)

    for i, basic_token in enumerate(basic_tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(basic_token)
        prob = random.random()

        if prob < 0.20 and drop:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    tokens.append(sub_token)
                    deleted_idx.append(len(tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                tokens.append(sub_token)

    if len(deleted_idx) != 0:
        tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]

    tokens = ['[CLS]'] + tokens + ['[SEP]']
    tokens = tokenizer.convert_tokens_to_ids(tokens)

    return torch.Tensor(tokens)


def process_caption(capenc_name, tool, caption, drop=False):
    if capenc_name in ['CapBERT']:
        return process_caption_bert(tool, caption, drop)
    else:
        return process_caption_bigru(tool, caption, drop)


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256) or (36, 1024).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256) or (batch_size, 36, 1024).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images
    images = torch.stack(images, 0)

    # Merget captions
    lengths = torch.LongTensor([len(cap) for cap in captions])
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids, img_ids
