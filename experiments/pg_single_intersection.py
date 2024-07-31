import argparse
import os
import sys
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


from sumo_rl import SumoEnvironment
from sumo_rl.agents import PolicyGradientAgent


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Policy Gradient RL Single-Intersection"""
    )

    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="sumo_rl/nets/single-intersection/single-intersection.rou.xml",
        help="Route definition xml file.\n",
    )


    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()

    experiment_time = str(datetime.now()).split(".")[0]
    out_csv = f"outputs/single-intersection/pg_final/{experiment_time}_alpha{args.alpha}_gamma{args.gamma}_eps{args.epsilon}_decay{args.decay}"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/single-intersection/single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
    )

    # start
    # running multiple episodes in the environment
    for run in range(1, args.runs + 1):
        # holds 11 features, including the 1-hot encoded value.
        initial_states = env.reset()
        print("just trying to debug here.")

        # create agent for each traffic light. There should be just one for our tests.
        # this loop should handle multiple. But we aren't going that far in this project.
        # NN is initialized here, too.
        pg_agents = {
            ts: PolicyGradientAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=args.alpha,
            )
            for ts in env.ts_ids
        }

        for ts_id, agent in pg_agents.items():
            print(agent.policy_nn)


        done = {"__all__": False}
        if args.fixed:
            while not done["__all__"]:
                _, _, done, _ = env.step({})


        env.save_csv(out_csv, run)
        env.close()
        # init time step counter
        # for each time step:
            # init s with current visual (state) of the intersection
            # select an action 'a' according to the policy
            # observe reward r and next state
            # compute gradients, according to equation 3 (referenced in paper)
            # update gradients according to equation 4 (referenced in paper)
    # until t - tstart == M (max time step)
    # end
