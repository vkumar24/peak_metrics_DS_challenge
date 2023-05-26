
class CleanString:
    def __init__(self):
        pass

    def lowercase_text(self, text):
        text = text.lower()
        return text

    def __call__(self, text):
        text = self.lowercase_text(text)
        return text