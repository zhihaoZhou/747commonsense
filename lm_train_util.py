import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LMTrainUtil:
    def __init__(self, train_iter, val_iter, model, device, config, vocab_size, TEXT):
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.SGD(model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config.milestones,
                                                        gamma=config.gamma)
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.model = model
        self.device = device
        self.config = config
        self.vocab_size = vocab_size
        self.TEXT = TEXT

    def train_epoch(self):
        self.scheduler.step()
        self.model.train()
        epoch_losses = []
        for batch in self.train_iter:
            x, y = batch.text.transpose(0, 1).contiguous().to(self.device), \
                   batch.target.transpose(0, 1).contiguous().to(self.device)

            self.optimizer.zero_grad()
            decoded, _, _ = self.model(x)

            loss = self.criterion(decoded.view(-1, self.vocab_size), y.view(-1))
            loss.backward()
            _ = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clipping)
            self.optimizer.step()

            epoch_losses.append(loss.item())
            # break
        return 2 ** np.mean(epoch_losses)

    def eval_epoch(self):
        self.model.eval()
        epoch_losses = []
        for batch in self.val_iter:
            x, y = batch.text.transpose(0, 1).contiguous().to(self.device), \
                   batch.target.transpose(0, 1).contiguous().to(self.device)

            with torch.no_grad():
                decoded, _, _ = self.model(x)
                loss = self.criterion(decoded.contiguous().view(-1, self.vocab_size),
                                      y.view(-1))
                epoch_losses.append(loss.item())
                # break
        return 2 ** np.mean(epoch_losses)

    def train_model(self):
        best_dev_perplex = float('inf')
        best_epoch = -1
        for epoch in range(self.config.num_epochs):
            cur_lr = self.optimizer.param_groups[0]['lr']
            train_perplex = self.train_epoch()
            dev_perplex = self.eval_epoch()

            if dev_perplex < best_dev_perplex:
                best_dev_perplex = dev_perplex
                best_epoch = epoch
                # early stopping
                torch.save(self.model.state_dict(), self.config.save_path)
                print('saved best model')

            print('epoch %d, lr %.5f, train_perplex %.4f, dev dev_perplex %.4f' %
                  (epoch, cur_lr, train_perplex, dev_perplex))
        print('best perplex %.4f, best epoch %d' % (best_dev_perplex, best_epoch))

        # # save most recent model regardless of dev score
        # torch.save(self.model.state_dict(), self.config.save_path)
        # print('saved best model')

    def load_trained_model(self):
        # load best model
        self.model.load_state_dict(torch.load(self.config.save_path))
        print('loaded best model')

    def predict(self, test_sentences):
        all_preds = []
        with torch.no_grad():
            decoded, outputs, hidden = self.model(test_sentences)
            # we only care about the last decoded
            last_decoded = decoded[:, -1, :]
            last_preds = last_decoded.argmax(1).unsqueeze(1)
            all_preds.append(last_preds)

            for i in range(60):
                decoded, outputs, hidden = self.model(last_preds, hidden)
                last_decoded = decoded[:, -1, :]
                last_preds = last_decoded.argmax(1).unsqueeze(1)
                all_preds.append(last_preds)

        all_preds = torch.cat(all_preds, dim=1)
        return all_preds

    @staticmethod
    def tokenizer(text):
        return text.split(" ")

    def generate(self):
        self.load_trained_model()
        # see some generations
        test_sentences_raw = ['I went into my bedroom and flipped the light switch',
                              'I got my keys and unlocked my car . I',
                              'I think it is time to do the laundry .',
                              'I was going to visit some friends in Florida .',
                              'I think the dentist is a little bit scary .']
        test_sentences = [self.tokenizer(sent) for sent in test_sentences_raw]
        test_sentences = [[self.TEXT.vocab.stoi[tok] for tok in sent] for sent in test_sentences]
        test_sentences = torch.LongTensor(test_sentences).to(self.device)

        # print('test_sentences', test_sentences.shape, test_sentences)
        all_preds = self.predict(test_sentences).cpu().numpy()
        all_preds = [[self.TEXT.vocab.itos[idx] for idx in row] for row in all_preds]
        all_preds = [' '.join(ele) for ele in all_preds]

        # visualize results
        for sent, pred in zip(test_sentences_raw, all_preds):
            print('%s ==> %s\n' % (sent, pred))
