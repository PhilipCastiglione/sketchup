class PredictionImageDetection:
    def __init__(self, x1, y1, x2, y2, label, score):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.label = label
        self.score = int(score)

