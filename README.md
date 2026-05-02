# Cross-Modal Refusal Vector Transfer

## Running on GCP VM

```bash
tmux new -s experiment
sudo apt install pip
pip install -r requirements.txt
python3 -c "from huggingface_hub import login; login()"
<<INSERT HF TOKEN>>
python3 <<filename.py>>
```

Relaunch tab:

```bash
tmux attach -t experiment
```
