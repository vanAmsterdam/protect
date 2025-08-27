"""
run this to generate the simulated data
"""
import pandas as pd
from jax import random
from protect.models import PROTECTModel
from protect.utils import load_yaml, time_event_to_time_cens

MAXTIME = 10
RNG_SEED = 123
NUM_OBS = [200, 500, 1000]

# load the numpyro model and its metadata
from model import tutorial_model
metadata = load_yaml('metadata.yaml')
metadata['local_prms'] = ['Feps']

model = PROTECTModel(tutorial_model, metadata, prior_spec="priors.csv")
model_control = metadata['model_control']

# load the simulation parameters
simdf = pd.read_csv('simprms.csv')
simprms = dict(zip(simdf.prm_name, simdf.value_sim))

def sim(rng_key, num_obs):
    data = model.simulate_from_prms(rng_key, simprms, num_obs, maxtime=MAXTIME, keep_deterministic=True, keep_locals=True)
    data['deceased'] = data['y'] < MAXTIME  
    data['time_cens'] = time_event_to_time_cens(data['y'], data['deceased'], maxtime=MAXTIME)

    # export
    df = pd.DataFrame(data)
    dropcols = ['b_tx_y_marginal']
    df = df.drop(columns=dropcols)

    # one version with the ground truth fhat and one without
    df.to_csv(f"data{num_obs}_full.csv", index=False)

    dropcols = ["Feps", "Fhat", "eta_tx", "lp", "lp_notx", "lp_dotx"]
    df.drop(columns=dropcols).to_csv(f"data{num_obs}.csv", index=False)

if __name__ == "__main__":
    rng_key = random.PRNGKey(RNG_SEED)
    for N in NUM_OBS:
        print(f"Simulating data for {N} observations...")

        # set the random seed for reproducibility
        rng_key, subkey = random.split(rng_key)
        sim(subkey, N)
    print("Data simulation complete.")