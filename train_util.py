import numpy as np
import torch
import sys
import time
import torch.nn as nn
import torch.optim as optim


class TrainUtil:
    @staticmethod
    def get_acc(outputs, labels, is_train=False):
        if is_train:
            preds = (outputs > 0.5).float()
            return np.mean((preds == labels).cpu().numpy().astype(float))
        else:
            outputs = outputs.cpu().numpy()
            preds = []
            for i in range(len(outputs)):
                if i % 2 == 0:
                    if outputs[i] > outputs[i + 1]:
                        preds += [1, 0]
                    else:
                        preds += [0, 1]
            preds = np.array(preds)
            labels = labels.cpu().numpy()
            return np.mean((preds == labels).astype(float))

    def __init__(self, train_iter, val_iter, model, device, config, TEXT):
        self.criterion = nn.BCELoss().to(device)
        self.optimizer = optim.Adamax(model.parameters(), lr=config.lr, weight_decay=0)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=config.milestones, gamma=config.gamma)
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.model = model
        self.device = device
        self.config = config
        self.TEXT = TEXT

    def calculate_lengths(self, words):
        """

        :param words: (batch_size, seq_len)
        :return: (batch_size,)
        """
        PAD_IDX = self.TEXT.vocab.stoi['<pad>']
        not_pad = words != PAD_IDX
        return torch.sum(not_pad, dim=1)

    def parse_batch(self, batch):
        d_words = batch.d_words
        q_words = batch.q_words
        c_words = batch.c_words
        d_words = torch.transpose(d_words, 0, 1)
        q_words = torch.transpose(q_words, 0, 1)
        c_words = torch.transpose(c_words, 0, 1)

        d_lengths = self.calculate_lengths(d_words)
        q_lengths = self.calculate_lengths(q_words)
        c_lengths = self.calculate_lengths(c_words)

        d_pos = batch.d_pos[0]
        d_ner = batch.d_ner[0]
        q_pos = batch.q_pos[0]

        in_q = batch.in_q[0].unsqueeze(dim=2).float()
        in_c = batch.in_c[0].unsqueeze(dim=2).float()
        lemma_in_q = batch.lemma_in_q[0].unsqueeze(dim=2).float()
        lemma_in_c = batch.lemma_in_c[0].unsqueeze(dim=2).float()
        tf = batch.tf[0].unsqueeze(dim=2).float()
        p_q_relation = batch.p_q_relation[0]
        p_c_relation = batch.p_c_relation[0]


        d_pos, d_ner, q_pos = torch.transpose(d_pos, 0, 1), \
                              torch.transpose(d_ner, 0, 1), torch.transpose(q_pos, 0, 1)

        in_q = torch.transpose(in_q, 0, 1)
        in_c = torch.transpose(in_c, 0, 1)
        lemma_in_q = torch.transpose(lemma_in_q, 0, 1)
        lemma_in_c = torch.transpose(lemma_in_c, 0, 1)
        tf = torch.transpose(tf, 0, 1)

        p_q_relation = torch.transpose(p_q_relation, 0, 1)
        p_c_relation = torch.transpose(p_c_relation, 0, 1)

        labels = batch.label.float()

        return d_words, d_pos, d_ner, d_lengths, q_words, q_pos, q_lengths, c_words, c_lengths, \
               labels, in_q, in_c, lemma_in_q, lemma_in_c, tf, p_q_relation, p_c_relation

    def train_epoch(self):
        self.scheduler.step()
        self.model.train()

        epoch_losses = []
        epoch_accus = 0.0
        total_num_example = 0

        for i, batch in enumerate(self.train_iter):
            # get batch
            d_words, d_pos, d_ner, d_lengths, q_words, q_pos, q_lengths, c_words, c_lengths, \
                labels, in_q, in_c, lemma_in_q, lemma_in_c, tf, p_q_relation, p_c_relation = self.parse_batch(batch)

            # get outputs and loss
            self.optimizer.zero_grad()
            outputs = self.model(d_words, d_pos, d_ner, d_lengths, q_words, q_pos, q_lengths, c_words, c_lengths, \
                            in_q, in_c, lemma_in_q, lemma_in_c, tf, p_q_relation, p_c_relation)
            loss = self.criterion(outputs, labels)

            # update model
            loss.to(self.device)
            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
            self.optimizer.step()

            # record losses and accuracies
            num_examples = d_words.shape[0]
            epoch_losses.append(loss.item())
            cur_acc = self.get_acc(outputs, labels, is_train=True)
            epoch_accus += cur_acc * num_examples
            total_num_example += num_examples
            # break

        accu_avg = epoch_accus / total_num_example
        loss_avg = np.mean(np.array(epoch_losses))

        return accu_avg, loss_avg

    def eval_epoch(self, debug=False):
        self.model.eval()

        total_num_example = 0
        epoch_losses = []
        epoch_accus = 0.0

        correct_labels = []
        prediction = []
        d_words_all = []
        q_words_all = []
        c_words_all = []

        # if debug:
        #     writer = open('data/analysis_d_q_c.log', 'w', encoding='utf-8')

        for i, batch in enumerate(self.val_iter):
            # get batch
            d_words, d_pos, d_ner, d_lengths, q_words, q_pos, q_lengths, c_words, c_lengths, \
            labels, in_q, in_c, lemma_in_q, lemma_in_c, tf, p_q_relation, p_c_relation = self.parse_batch(batch)

            correct_labels += [float(label) for label in labels]
            d_words_all.append(d_words)
            q_words_all.append(q_words)
            c_words_all.append(c_words)

            # eval
            with torch.no_grad():
                outputs = self.model(d_words, d_pos, d_ner, d_lengths, q_words, q_pos, \
                                q_lengths, c_words, c_lengths, in_q, in_c, lemma_in_q, lemma_in_c, tf, p_q_relation,
                                p_c_relation)
                prediction += [float(output) for output in outputs]

                loss = self.criterion(outputs, labels)
                # record losses and accuracies
                num_examples = d_words.shape[0]
                epoch_losses.append(loss.item())
                cur_acc = self.get_acc(outputs, labels, is_train=False)
                epoch_accus += cur_acc * num_examples
                total_num_example += num_examples
                # break

        accu_avg = epoch_accus / total_num_example
        loss_avg = np.mean(np.array(epoch_losses))

        return accu_avg, loss_avg

    def train_model(self):
        # writer_acc = open('data/analysis_accs.log', 'w', encoding='utf-8')

        # training loop
        for epoch in range(self.config.num_epoch):
            print('~' * 80)

            train_accs = []
            eval_accs = []

            cur_lr = self.optimizer.param_groups[0]['lr']
            start = time.time()

            train_accu, train_loss = self.train_epoch()

            eval_accu, eval_loss = self.eval_epoch(debug=True)

            # writer_acc.write('Epoch: {} \n'.format(str(epoch)))
            # writer_acc.write('Train accuracy: %.4f    Eval accuracy: %.4f \n' % (train_accu, eval_accu))

            train_accs.append(train_accu)
            eval_accs.append(eval_accu)

            end = time.time()
            print('%dth iteration took %.4f' % (epoch, end - start))
            print('learning rate: %.4f' % cur_lr)
            print('train_loss: %.4f' % train_loss)
            print('eval_loss: %.4f' % eval_loss)
            print('train_accuracy: %.4f' % train_accu)
            print('eval_accuracy: %.4f' % eval_accu)
            sys.stdout.flush()
