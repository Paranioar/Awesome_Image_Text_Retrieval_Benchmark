import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import ConcatAttnLayer

import logging
logger = logging.getLogger(__name__)


class EncoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim,
                 inp_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False):

        super(EncoderRNN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.inp_dropout_p = inp_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers

        self.emb2hid = nn.Linear(embed_dim, hidden_dim)
        self.input_dropout = nn.Dropout(inp_dropout_p)

        self.rnn = self.nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True,
                               bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.emb2hid.weight)

    def forward(self, input):

        nbatch, ninstance, ndim = input.size()
        input = self.emb2hid(input.view(-1, ndim))
        input = self.input_dropout(input)
        input = input.view(nbatch, ninstance, self.hidden_dim)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(input)
        return output, hidden


class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, hidden_dim, word_dim,
                 inp_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, max_len=28, bidirectional=False):
        super(DecoderRNN, self).__init__()

        self.bidirectional = bidirectional
        self.output_dim = vocab_size
        self.hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.word_dim = word_dim
        self.max_length = max_len
        self.sos_id = 1
        self.eos_id = 0
        self.input_dropout = nn.Dropout(inp_dropout_p)
        self.embedding = nn.Embedding(self.output_dim, word_dim)
        self.attention = ConcatAttnLayer(self.hidden_dim)

        self.rnn = nn.GRU(self.hidden_dim + word_dim, self.hidden_dim,
                          n_layers, batch_first=True, dropout=rnn_dropout_p)

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        self._init_weights()

    def forward(self, encoder_outputs, encoder_hidden, targets, mode='train'):
        opt = {}
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []
        self.rnn.flatten_parameters()
        if mode == 'train':
            # use targets as rnn inputs
            targets_emb = self.embedding(targets)
            for i in range(self.max_length - 1):
                current_words = targets_emb[:, i, :]
                context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs.unsqueeze(1))

            seq_logprobs = torch.cat(seq_logprobs, 1)

        elif mode == 'inference':
            if beam_size > 1:
                return self.sample_beam(encoder_outputs, decoder_hidden, opt)

            for t in range(self.max_length - 1):
                context = self.attention(
                    decoder_hidden.squeeze(0), encoder_outputs)

                if t == 0:  # input <bos>
                    it = torch.LongTensor([self.sos_id] * batch_size).cuda()
                elif sample_max:
                    sampleLogprobs, it = torch.max(logprobs, 1)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()

                else:
                    # sample according to distribuition
                    if temperature == 1.0:
                        prob_prev = torch.exp(logprobs)
                    else:
                        # scale logprobs by temperature
                        prob_prev = torch.exp(torch.div(logprobs, temperature))
                    it = torch.multinomial(prob_prev, 1).cuda()
                    sampleLogprobs = logprobs.gather(1, it)
                    seq_logprobs.append(sampleLogprobs.view(-1, 1))
                    it = it.view(-1).long()

                seq_preds.append(it.view(-1, 1))

                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(
                    decoder_input, decoder_hidden)
                logprobs = F.log_softmax(
                    self.out(decoder_output.squeeze(1)), dim=1)

            seq_logprobs = torch.cat(seq_logprobs, 1)
            seq_preds = torch.cat(seq_preds[1:], 1)

        return seq_logprobs, seq_preds

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


class S2VTModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_dim=512, word_dim=512,
                 max_len=28, sos_id=1, eos_id=0, n_layers=1, rnn_dropout_p=0.2):
        super(S2VTModel, self).__init__()

        self.rnn1 = nn.GRU(embed_size, hidden_dim, n_layers,
                           batch_first=True, dropout=rnn_dropout_p)
        self.rnn2 = nn.GRU(hidden_dim + word_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=rnn_dropout_p)

        self.embed_dim = embed_size
        self.output_dim = vocab_size
        self.hidden_dim = hidden_dim
        self.word_dim = word_dim
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.output_dim, self.word_dim)

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, target, mode='train'):
        batch_size, n_frames, _ = input.shape
        padding_words = input.data.new(batch_size, n_frames, self.word_dim).zero_()
        padding_frames = input.data.new(batch_size, 1, self.embed_dim).zero_()
        state1 = None
        state2 = None
        output1, state1 = self.rnn1(input, state1)
        input2 = torch.cat((output1, padding_words), dim=2)
        output2, state2 = self.rnn2(input2, state2)

        seq_probs = []
        seq_preds = []
        if mode == 'train':
            for i in range(self.max_length - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target[:, i])
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)

        else:
            current_words = self.embedding(
                torch.LongTensor([self.sos_id] * batch_size).cuda())
            for i in range(self.max_length - 1):
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds


class S2VTAttnModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024,
                 hidden_dim=512, word_dim=512,
                 inp_dropout_p=0.2, rnn_dropout_p=0.5,
                 max_len=28, n_layers=1,
                 bidirectional=False):
        super(S2VTAttnModel, self).__init__()

        self.encoder = EncoderRNN(embed_dim=embed_size,
                                  hidden_dim=hidden_dim,
                                  inp_dropout_p=inp_dropout_p,
                                  rnn_dropout_p=rnn_dropout_p,
                                  n_layers=n_layers,
                                  bidirectional=bidirectional)

        self.decoder = DecoderRNN(vocab_size=vocab_size,
                                  hidden_dim=hidden_dim,
                                  word_dim=word_dim,
                                  inp_dropout_p=inp_dropout_p,
                                  rnn_dropout_p=rnn_dropout_p,
                                  n_layers=n_layers,
                                  max_len=max_len,
                                  bidirectional=bidirectional)

    def forward(self, input, target, mode='train'):
        """
        Args:
            input (Variable): input shape [batch_size, seq_len, embed_size]
            target (None, optional): groung truth labels
        Returns:
            seq_prob: Variable of shape [batch_size, max_len-1, vocab_size]
            seq_preds: [] or Variable of shape [batch_size, max_len-1]
        """
        output, hidden = self.encoder(input)
        seq_prob, seq_preds = self.decoder(output, hidden, target, mode)
        return seq_prob, seq_preds