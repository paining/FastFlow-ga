import torch
import numpy as np
from tqdm import tqdm
from typing import Union

def coreset_samping(
    feature_array:torch.Tensor,
    sampling_number:int,
    device:Union[str, int, torch.device, None]=None,
    min_distance:Union[int, None]=None,
) -> (list, float):
    if min_distance is None:
        min_distance = 0
    device = torch.device(device)
    N = feature_array.shape[0]
    if not isinstance(feature_array, torch.Tensor):
        feature_array = torch.tensor(feature_array)

    feature_array = feature_array.to(device)
    min_distances = torch.Tensor(size=(N,0)).to(device)
    cluster_centers = []
    np.random.seed(0)
    ind = np.random.choice(np.arange(N))

    for i in tqdm(
        range(sampling_number),
        ncols=79,
        desc="|coreset| Sampling...",
        leave=False,
        dynamic_ncols=True
    ):
        cluster_centers.append(ind)
        x = feature_array[ind,:] # shape = (i, D)
        # calculate distance from samples to center of clusters.
        dist = torch.cdist(
            feature_array.unsqueeze(0), x.unsqueeze(0)
            ).squeeze(0) # dist.shape = (N, i)

        # calculate minimum distance from center of clusters.
        min_distances = torch.min( 
            torch.concat([min_distances, dist], dim=1), 
            dim=1, keepdim=True
        ).values

        # find maximum distance sample.
        ind = torch.argmax(min_distances).item()
        max_distance = min_distances[ind].item()

        if max_distance < min_distance:
            tqdm.write(f"|coreset| reach to min_distance({min_distance})")
            break

    return (cluster_centers, max_distance)