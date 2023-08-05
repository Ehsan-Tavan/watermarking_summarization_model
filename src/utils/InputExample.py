from typing import Union, List


class InputExample:
    def __init__(self, text: str, summary: str = None):
        self.text = text
        self.summary = summary

    def __str__(self):
        return "<InputExample> text: {}, summary: {}".format(str(self.summary), str(self.text))
