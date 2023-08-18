# InstaSynth
Synthetic Instagram Post Generation for Social Media Research

### Setting up the Repository
1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/thalesbertaglia/instasynth.git
   cd instasynth
   ```

2. **Install Dependencies using Poetry:**  
   ```bash
   poetry install
   ```

   This command will read the `pyproject.toml` file from the current project, resolve the dependencies and install them.

3. **Activate the Poetry Environment:**  
   ```bash
   poetry shell
   ```

   This will spawn a shell within the virtual environment.


4. **Add your OpenAI API key to the .env file**
   ```bash
   OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   ```

If you're using Jupyter:

5. **Start Jupyter Notebook:**  
   ```bash
   poetry run jupyter notebook
   ```

   This will launch the Jupyter Notebook, and you can navigate to the desired `.ipynb` file to use it.

If you're using another notebook tool, you can usually start it within the activated Poetry shell.
