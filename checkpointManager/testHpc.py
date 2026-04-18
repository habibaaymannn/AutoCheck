import time
import random
from pathlib import Path

from checkpointManager.GenericCheckpointManager import GenericCheckpointManager
from checkpointManager.serializers import JobLibSerializer, JSONSerializer


SAVE_DIR = "hpc_checkpoints"


class HPCSimulation:
    def __init__(self, manager: GenericCheckpointManager):
        self.manager = manager

        # state
        self.iteration = 0
        self.partial_sum = 0.0
        self.data_buffer = []

    def get_state(self):
        return {
            "iteration": self.iteration,
            "partial_sum": self.partial_sum,
            "data_buffer": self.data_buffer,
        }

    def load_state(self, state):
        self.iteration = state["iteration"]
        self.partial_sum = state["partial_sum"]
        self.data_buffer = state["data_buffer"]

    def compute_step(self):
        # simulate heavy computation
        value = random.random()
        self.partial_sum += value
        self.data_buffer.append(value)

        # keep buffer limited (like real HPC memory control)
        if len(self.data_buffer) > 100:
            self.data_buffer.pop(0)

        self.iteration += 1

    def run(self, max_iters=50, checkpoint_interval=10, simulate_crash_at=25):

        print(f"🚀 Starting HPC simulation (max_iters={max_iters})")

        # 🔄 Try resume
        try:
            state = self.manager.load_checkpoint(SAVE_DIR)
            self.load_state(state)
            print(f"🔁 Resumed from iteration {self.iteration}")
        except Exception:
            print("🆕 No checkpoint found, starting fresh")

        while self.iteration < max_iters:

            self.compute_step()

            print(f"Iteration {self.iteration} | sum={self.partial_sum:.4f}")

            # 💾 checkpoint
            if self.iteration % checkpoint_interval == 0:
                self.manager.save_checkpoint(self.get_state(), SAVE_DIR)
                print(f"💾 Checkpoint saved at iteration {self.iteration}")

            # 💥 simulate HPC preemption / crash
            if self.iteration == simulate_crash_at:
                print("💥 Simulated job interruption!")
                return  # exit early like HPC job killed

            time.sleep(0.1)

        print("✅ Simulation completed successfully!")


# =========================
# RUN TEST
# =========================

if __name__ == "__main__":

    manager = GenericCheckpointManager(
        checkpoint_dir=SAVE_DIR,
        serializer=JSONSerializer(),
        max_to_keep=3,
    )

    sim = HPCSimulation(manager)

    # First run (will crash)
    sim.run(max_iters=50, checkpoint_interval=10, simulate_crash_at=25)

    print("\n🔄 Restarting job...\n")

    # Second run (should resume)
    sim2 = HPCSimulation(manager)
    sim2.run(max_iters=50, checkpoint_interval=10, simulate_crash_at=999)