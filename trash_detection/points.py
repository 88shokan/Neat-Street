class User:
    def __init__(self):
        self.labels = set()  # Using a set to store unique labels
        self.score = 0  # Initial score

    def add_label(self, label):
        if label not in self.labels:
            self.labels.add(label)
            print(f"New label detected and added: {label}")

    def detect_trash(self, final_label):
        if final_label != "no labels":  # Check if a valid label was detected
            self.add_label(final_label)
            self.score += 5  # Award points
            print(f"Points awarded! Current score: {self.score}")

