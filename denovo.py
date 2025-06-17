from gradio_client import Client
import pandas as pd
import re

def generate_protein(prompt,num):
  client = Client("http://www.denovo-pinal.com/")
  result = client.predict(
      input=prompt,
      designed_num=num,
      api_name="/design_and_protrek_score"
     )
  # Step 1: Extract rows from the markdown table using regex
  rows = re.findall(r'\|\s*(\d+)\s*\|\s*(-?\d*\.\d+)\s*\|\s*(\d*\.\d+)\s*\|\s*(.*?)\s*\|', result[0])

  # Step 2: Convert to DataFrame
  df = pd.DataFrame(rows, columns=["Index", "LogPPerToken", "ProtrekScore", "ProteinSequence"])

  # Step 3: Type conversion
  df["Index"] = df["Index"].astype(int)
  df["LogPPerToken"] = df["LogPPerToken"].astype(float)
  df["ProtrekScore"] = df["ProtrekScore"].astype(float)
  df = df.sort_values(by="ProtrekScore", ascending=False)
  df = df.reset_index(drop=True)


  return df