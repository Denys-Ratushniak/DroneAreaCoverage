from environment import DroneEnv
from rectangle import Rectangle

if __name__ == "__main__":
    env = DroneEnv(
        n_drones=1,
        delta_time=0.07,
        render_mode=DroneEnv.human,
        rectangle=Rectangle((0, 0), (1000, 1000), (50, 50))
    )

    env.reset()
    env.run()
