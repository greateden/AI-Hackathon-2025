import pandas as pd, os
csv = os.path.expanduser("~/.cache/kagglehub/datasets/equintel/dax-esg-media-dataset/versions/3/esg_documents_for_dax_companies.csv")
cols = pd.read_csv(csv, engine="pyarrow", nrows=0).columns.tolist()
print("Columns:", cols)
print("Has 'content'? ->", "content" in cols)
