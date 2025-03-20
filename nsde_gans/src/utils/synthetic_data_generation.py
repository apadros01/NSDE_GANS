import torch
import numpy as np
import pandas as pd


def sample_generator(generator,ts):
    samples = generator(ts, batch_size=1)
    samples = samples[0]
    samples = torch.abs(samples[-1])
    return samples.detach().cpu().numpy()

def synthtetic_dataset(generator,ts,df, size=3): # by default, df = df_no_cvd.iloc[:,1:27]
    samples=[]
    for i in range(size):
        samples.append(sample_generator(generator,ts))
    dades=np.array(samples)
    return pd.DataFrame(data=dades, columns=df.columns)