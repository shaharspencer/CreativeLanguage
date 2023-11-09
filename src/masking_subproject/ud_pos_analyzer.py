

from datasets import load_dataset



class EvaluateSpacy:
    def __init__(self):
        self.dataset = load_dataset("universal_dependencies", "en_gum")


if __name__ == '__main__':
    pass