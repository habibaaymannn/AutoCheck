import json
from pathlib import Path
from datetime import datetime


class StateStore:# handels the saving and loading of the state snapshots to and from disk
    

    def __init__(self, directory="checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    def save(self, snapshot: dict, filename=None):
    
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # if the file name is null, we generate a unique name based on the current timestamp
            filename = f"checkpoint_{timestamp}.json"

        path = self.directory / filename

        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

        return path

    # Load cehckpoint
    def load(self, filename): # load a specific checkpoint if we need a specific one else we load the latest one
        path = self.directory / filename

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, "r") as f:
            return json.load(f)


    # Load Latest (most recent) Checkpoint
    def load_latest(self):
        
        files = sorted(self.directory.glob("checkpoint_*.json"))
        if not files:
            raise FileNotFoundError("No checkpoints found")

        latest = files[-1]
        with open(latest, "r") as f:
            return json.load(f)

    # List of Checkpoints
    def list_checkpoints(self):
        return sorted([p.name for p in self.directory.glob("checkpoint_*.json")])
