# Documentation

## Setup

### Mount the Remote Directory Locally *

Since running the code requires a GPU, the code must already be on the GPU server. If you are comfortable with tools like `vim`, that’s fine. But if you prefer using Visual Studio Code (VSCode), there is a way to mount your remote directory locally in VSCode. This allows you to modify the code on your local editor, and the changes will be reflected remotely. You can then run the code directly on the GPU server.

First, you need to install the extension: **Remote - SSH**.

![Remote SSH Extension in VSCode](remotessh.png)

Once the extension is installed, you need to add the remote machine. Click the **+** button, as shown in the image below:

![Add a computer](add.png)

Type `username@lig-potato.imag.fr` and then specify the directory on the remote machine that you want to mount (e.g., `subjective-comment-clf` in our case). (You have to be connected on the VPN).
You will be asked to specify the os of the GPU, it is Linux.

Now open the VSCode terminal. In the next section, we will see how to connect to the GPU and run the code.

### Run Code on the GPU

First, check for an available GPU on the website: [http://aker.imag.fr/drawgantt/](http://aker.imag.fr/drawgantt/). You should be connected on the VPN.

Once a GPU is available, run the following command in your terminal:

```bash
ssh username@aker.imag.fr
```

Aker is an entry point server used for scheduling GPU jobs. Then type:

```bash
oarsub -I -p "host='lig-gpu10.imag.fr'" -l /gpu=1,walltime=7:0:0
```

Here's a breakdown of the command:

| Command | Description |
|---------|-------------|
| `oarsub` | Submits a job to the OAR job scheduler. |
| `-I` | Starts an interactive session (so you get direct access to the GPU machine). |
| `-p "host='lig-gpu10.imag.fr'"` | Requests a specific host (in this case, `lig-gpu10.imag.fr`). |
| `-l /gpu=1` | Requests one GPU. |
| `walltime=7:0:0` | Sets a maximum job duration of 7 hours. |

After submitting the command, wait until the job starts. Once connected, you’ll be on the GPU server and ready to run your code.

### Load the Code from GitHub *

Once connected to the GPU server, you can directly load the code from GitHub. Since your GPU home directory is usually empty on first connection, it’s useful to create a folder structure that mirrors your local environment. For example, you can start by creating a `Documents` folder:

```bash
mkdir Documents
```

Then, move into that folder:

```bash
cd Documents
```

Now, clone the GitHub repository:

```bash
git clone https://github.com/kmadou019/subjective-comment-clf.git
```

### Install a Python Environment *

Go to the `subjective-comment-clf` folder to create the environment.
Create a virtual environment by running the following command:

```bash
python3 -m venv mon_env/
```

After creating the environment, you need to activate it. From the folder that contains `mon_env/`, run the following command:

```bash
source mon_env/bin/activate
```

### Install Requirements *

Running the code requires specific libraries and tools. First, navigate to the `subjective-comment-clf` folder. There you’ll find a file named `requirements.txt`, which lists all the necessary libraries.

To install them, run:

```bash
pip install -r requirements.txt
```

### Install Ollama (Linux - Without Root Access) *

We need to install **Ollama**. The official installer can be found here: [https://ollama.com/download/linux](https://ollama.com/download/linux).

However, the installation command on the Ollama website requires root access, which is not available for custom users on the GPU server. Therefore, we need to install it [manually](https://github.com/ollama/ollama/blob/main/docs/linux.md).

Follow these steps:

1. Go to your home directory:

```bash
cd ~
```

2. Create a new folder where Ollama will be installed:

```bash
mkdir ollama
```

3. Download the compressed Ollama file:

```bash
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
```

4. Extract it into the folder you just created:

```bash
tar -C ./ollama -xzvf ollama-linux-amd64.tgz
```

5. Add Ollama to your `PATH` by appending the following line to your `~/.bashrc` file:

```bash
echo "export PATH=$PATH:$HOME/ollama/bin" >> ~/.bashrc
```

6. Reload your `~/.bashrc` so the changes take effect:

```bash
source ~/.bashrc
```

7. Check if Ollama is properly installed by running:

```bash
ollama ls
```

#### Download the Required Models

Our implementation uses three models. Download them using the following commands:

```bash
ollama serve & (then Ctrl + C)
```

```bash
ollama pull mistral
```

```bash
ollama pull phi4
```

```bash
ollama pull llama3.3
```

### Add a Hugging Face Token for ChromaDB *

First, go to [https://huggingface.co/](https://huggingface.co/) and create an account if you don't already have one. Then, click on your profile picture at the top right of the page and select **Access Tokens** as shown below:

![Menu to access Hugging Face tokens](access-token.png)

Click the **Create new token** button:

![Create new token button](create-token.png)

Then, generate a new token and make sure the token type is set to **write**:

![Selecting 'write' as token type](write.png)

Save the generated token. We will refer to it as `hf_token` in the following steps.

Now, open your terminal and run the following command to store the token as an environment variable:

```bash
echo "export HUGGINGFACE_API_KEY='hf_token'" >> ~/.bashrc
```

Then, reload your `~/.bashrc` to apply the changes:

```bash
source ~/.bashrc
```

## Architecture

The general architecture of the code is as follows:

![General architecture](structure.png)

| Folder/File | Description |
|-------------|-------------|
| **data/** | This folder contains two CSV files. `comments.csv` contains the comments used to populate the database, and `test.csv` contains the comments used to evaluate the system. |
| **excel/** | Contains the output files generated by the system. |
| **script/** | Contains the script used to execute the code on the test set. |
| **chroma.py** | This script creates the database from `data/comments.csv`. After execution, it generates a folder named `chroma_db/`. |

### Alter the Database

To modify or completely replace the database, edit the file `data/comments.csv`. Make sure that the modified file respects the expected format:

![Training set format example](trainset.png)

After making your changes, you must regenerate the database by running the script below. Make sure you are in the `subjective-comment-clf/rag` folder, which contains `chroma.py`:

```bash
./chroma.py
```

### Modify Categories and Keywords

To change the categories or keywords used by the system, open the file `graph_builder.py` and modify the corresponding prompt section.

![Prompt section in graph_builder.py](prompt.png)

### Modify the Number of Retrieved Documents

To change `k`, the number of documents retrieved during inference, open the file `graph_builder.py` and locate the line that sets `k = 10` as shown in the figure below. Then put the value you want.

![Modification of k in graph_builder.py](k.png)

### Copy Modifications to GPU

Since there is not a direct synchronization between the CPU and GPU servers, you need to manually copy the modified files from the CPU server (e.g., **potato**) to the GPU server (e.g., **aker**). Make sure you are connected to the **potato** server (not **aker**) before running the command below:

```bash
scp -r ~/Documents/subjective-comment-clf/rag username@aker.imag.fr:~/Documents/subjective-comment-clf/
```

### Run the Code

The file `rag.sh` allows you to run the code on the test set. It contains the following three lines:

![Contents of rag.sh](run.png)

Each line corresponds to the execution of the system with a different model. By default, only the first line (using **Mistral**) is active — the other two are commented out.

To switch models, simply uncomment the line corresponding to the desired model and comment out the others. Make sure you are in the `subjective-comment-clf/rag/script` folder. Then, run the script with:

```bash
./rag.sh
```

### Transfer the Output

The output files generated by the system are located in the `subjective-comment-clf/excel/` folder. Since the code was executed on the GPU server, this folder is stored on the GPU machine. You need to copy it from the GPU server to the CPU server (e.g., **potato**). Make sure you are connected to the **potato** server (not **aker**) before running the command below (don't forget the . at the end and replace `username` with your actual username):

```bash
scp -r username@aker.imag.fr:~/Documents/subjective-comment-clf/excel .
```