
import re
import pandas as pd

class TextPreprocessor:
    def clean_text(self, text):
        text = str(text)
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "title" not in out.columns:
            raise ValueError("Expected a title column in the input dataframe.")
        out["title"] = out["title"].fillna("").astype(str)
        out["clean_title"] = out["title"].apply(self.clean_text)
        return out
