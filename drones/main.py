from environment import DroneEnv
from rectangle import Rectangle
from models import *
import torch


if __name__ == "__main__":
    run_random = True

    if run_random:
        env = DroneEnv(
            n_drones=2,
            delta_time=0.07,
            render_mode=DroneEnv.human,
            rectangle=Rectangle((0, 0), (1000, 1000), (50, 50))
        )

        env.reset()
        env.run()
    else:
        env = DroneEnv(
            n_drones=1,
            delta_time=0.07,
            render_mode=DroneEnv.human,
            rectangle=Rectangle((0, 0), (1000, 1000), (50, 50))
        )

        # Path to the saved weights
        actor_path = 'actor_model.pth'
        critic_path = 'critic_model.pth'

        env.reset_and_run_with_models(actor_path, critic_path)

