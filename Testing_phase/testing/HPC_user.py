# user_hpc.py – simulation that updates a global variable 'step'
class Simulation:
    def __init__(self):
        self.current_step = 0

    def run(self):
        for _ in range(1000):
            self.current_step += 1
            # simulate work
            time.sleep(0.01)


simulation = Simulation()
simulation.run()
