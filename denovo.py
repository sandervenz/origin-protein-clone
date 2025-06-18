from gradio_client import Client
import pandas as pd
import re
import httpx  
import streamlit as st 

def generate_protein(prompt,num):
    try:
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
    
    except httpx.ReadTimeout:
        st.error("Connection to sequence generation server (denovo-pinal.com) timed out. The server is likely down or very slow. Please try again later.")
        return pd.DataFrame() 

    except Exception as e:
        st.error(f"An unexpected error occurred with the sequence generator: {e}")
        return pd.DataFrame() 