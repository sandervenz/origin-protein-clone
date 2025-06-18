# Step 1: Use an official Miniconda3 image as a base
FROM continuumio/miniconda3:latest

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the environment file to the working directory
COPY environment.yml .

# Step 4: Create the Conda environment from the YAML file
# This will install all dependencies into an environment named 'protein_env'
RUN conda env create -f environment.yml

# Step 5: Copy all your project files into the container
COPY . .

# Step 6: Expose the default Streamlit port
EXPOSE 8501

# Step 7: Define the entrypoint using the absolute path to the executable in the new environment
ENTRYPOINT ["/opt/conda/envs/protein_env/bin/streamlit", "run"]

# Step 8: Define the default arguments for the entrypoint
CMD ["app.py", "--server.port=8501", "--server.address=0.0.0.0"]