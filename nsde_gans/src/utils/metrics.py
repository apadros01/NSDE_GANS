#https://arxiv.org/pdf/2104.00635.pdf
import pandas as pd
import numpy as np
import torch

def bin_data(dt1, dt2, c = 10):
    dt1 = dt1.copy()
    dt2 = dt2.copy()
    # quantile binning of numerics
    num_cols = dt1.dtypes[dt1.dtypes!='object'].index
    for col in num_cols:
        # determine breaks based on `dt1`
        breaks = dt1[col].quantile(np.linspace(0, 1, c+1)).unique()
        dt1[col] = pd.cut(dt1[col], bins=breaks, include_lowest=True).astype(str)
        dt2_vals = pd.to_numeric(dt2[col], 'coerce')
        dt2_bins = pd.cut(dt2_vals, bins=breaks, include_lowest=True).astype(str)
        dt2_bins[dt2_vals < min(breaks)] = '_other_'
        dt2_bins[dt2_vals > max(breaks)] = '_other_'
        dt2[col] = dt2_bins
    # top-C binning of categoricals
    cat_cols = dt1.dtypes[dt1.dtypes=='object'].index
    for col in cat_cols:
        # determine top values based on `dt1`
        top_vals = dt1[col].value_counts().head(c).index.tolist()
        dt1[col].replace(np.setdiff1d(dt1[col].unique().tolist(), top_vals), '_other_', inplace=True)
        dt2[col].replace(np.setdiff1d(dt2[col].unique().tolist(), top_vals), '_other_', inplace=True)
    return [dt1, dt2]

def hellinger(p1, p2):
    return np.sqrt(1 - np.sum(np.sqrt(p1*p2)))

def kullback_leibler(p1, p2):
    idx = p1>0
    return np.sum(p1[idx] * np.log(p1[idx]/p2[idx]))

def jensen_shannon(p1, p2):
    m = 0.5 * (p1 + p2)
    return 0.5 * kullback_leibler(p1, m) + 0.5 * kullback_leibler(p2, m)

def fidelity(dt1, dt2, c = 100, k = 1):
    [dt1_bin, dt2_bin] = bin_data(dt1, dt2, c = c)
    # build grid of all cross-combinations
    cols = dt1.columns
    interactions = pd.DataFrame(np.array(np.meshgrid(cols, cols, cols)).reshape(3, len(cols)**3).T)
    interactions.columns = ['dim1', 'dim2', 'dim3']
    if k == 1:
        interactions = interactions.loc[(interactions['dim1']==interactions['dim2']) & (interactions['dim2']==interactions['dim3'])]
    elif k == 2:
        interactions = interactions.loc[(interactions['dim1']<interactions['dim2']) & (interactions['dim2']==interactions['dim3'])]
    elif k == 3:
        interactions = interactions.loc[(interactions['dim1']<interactions['dim2']) & (interactions['dim2']<interactions['dim3'])]
    else:
        raise('k>3 not supported')

    results = []
    for idx in range(interactions.shape[0]):
        row = interactions.iloc[idx]
        val1 = dt1_bin[row.dim1] + dt1_bin[row.dim2] + dt1_bin[row.dim3]
        val2 = dt2_bin[row.dim1] + dt2_bin[row.dim2] + dt2_bin[row.dim3]
        freq1 = val1.value_counts(normalize=True).to_frame(name='p1')
        freq2 = val2.value_counts(normalize=True).to_frame(name='p2')
        freq = freq1.join(freq2, how='outer').fillna(0.0)
        p1 = freq['p1']
        p2 = freq['p2']
        out = pd.DataFrame({
          'k': k,
          'dim1': [row.dim1], 'dim2': [row.dim2], 'dim3': [row.dim3],
          'tvd': [np.sum(np.abs(p1 - p2)) / 2], 
          'mae': [np.mean(np.abs(p1 - p2))], 
          'max': [np.max(np.abs(p1 - p2))],
          'l1d': [np.sum(np.abs(p1 - p2))],
          'l2d': [np.sqrt(np.sum((p1 - p2)**2))],
          'hellinger': [hellinger(p1, p2)],
          'jensen_shannon': [jensen_shannon(p1, p2)]})
        results.append(out)

    return pd.concat(results)


def evaluate_loss(ts, batch_size, dataloader, generator, discriminator):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples