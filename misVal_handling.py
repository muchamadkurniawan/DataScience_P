import pandas as pd

class missingValue:
    df = []

    def __init__(self, dataFrame):
        self.df = dataFrame

    def dropMissVal(self):
        self.df = self.df.dropna()

