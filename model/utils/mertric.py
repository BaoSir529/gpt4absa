class Mertics:
    def __init__(self):
        self.true_set = set()
        self.pred_set = set()

    def pred_inc(self, idx, preds):
        for pred in preds:
            self.pred_set.add((idx, pred.lower()))

    def true_inc(self, idx, trues):
        for true in trues:
            self.true_set.add((idx, true.lower()))

    def report(self):
        self.f1, self.p, self.r = self.cal_f1(self.pred_set, self.true_set)
        return self.f1

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise NotImplementedError

    def cal_f1(self, pred_set, true_set):
        intersection = pred_set.intersection(true_set)
        _p = len(intersection) / len(pred_set) if pred_set else 1
        _r = len(intersection) / (len(true_set)) if true_set else 1
        f1 = 2 * _p * _r / (_p + _r) if _p + _r else 0
        return f1, _p, _r