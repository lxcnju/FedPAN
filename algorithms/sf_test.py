import torch
import copy
import numpy as np
from utils import count_acc, Averager
from utils import append_to_logs
from utils import format_logs


class ShuffleTest():
    def __init__(
        self,
        test_set,
        model,
        args
    ):
        self.test_set = test_set
        self.model = model
        self.args = args

        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=128, shuffle=False
        )

        self.sf_ratios = [
            0.0, 0.01, 0.02, 0.03, 0.05,
            0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0
        ]

        self.sf_probs = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 1.0
        ]

        self.logs = {
            "SFRATIOS": [],
            "SFPROBS": [],
            "SFACCS": [],
        }

    def main(self):
        raw_te_acc = self.test(
            model=self.model,
            loader=self.test_loader,
            args=self.args
        )

        for sf_ratio in self.sf_ratios:
            for sf_prob in self.sf_probs:
                sf_te_accs = []
                for k in range(3):
                    model = copy.deepcopy(self.model)
                    model.shuffle(
                        sf_ratio=sf_ratio,
                        sf_prob=sf_prob,
                        reshuffle=True
                    )

                    sf_te_acc = self.test(
                        model=model,
                        loader=self.test_loader,
                        args=self.args
                    )
                    sf_te_accs.append(sf_te_acc)

                sf_te_acc = np.mean(sf_te_accs)
                print("[#ShuffleTest:{},{}] [{:.4f},{:.4f}]".format(
                    sf_ratio, sf_prob, raw_te_acc, sf_te_acc
                ))

                self.logs["SFRATIOS"].append(str(sf_ratio))
                self.logs["SFPROBS"].append(str(sf_prob))
                self.logs["SFACCS"].append(sf_te_acc)

    def test(self, model, loader, args):
        model.eval()

        acc_avg = Averager()
        with torch.no_grad():
            for tx, ty in loader:
                if args.cuda:
                    tx, ty = tx.cuda(), ty.cuda()

                _, logits = model(tx)

                acc = count_acc(logits, ty)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_ckpt(self, fpath):
        # save model
        torch.save(self.model.state_dict(), fpath)
        print("Model saved in: {}".format(fpath))

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
