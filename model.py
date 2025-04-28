import pandas as pd

class AllergenModel:
    def __init__(self, df):
        self.df = df

    def search(self, query):
        query = query.lower()
        result = self.df[
            self.df['food'].str.contains(query) |
            self.df['type'].str.contains(query) |
            self.df['group'].str.contains(query) |
            self.df['class'].str.contains(query) |
            self.df['allergy'].str.contains(query)
        ]
        if result.empty:
            return "❌ No allergen data found for that query."
        return result.reset_index(drop=True)