import os, pickle
import numpy as np
from pp_netlib.network import Network


def network_from_sketchlib(data_dir, backend, outdir = "./", weights_type = "core"):
    for sketchlib_out in os.listdir(data_dir):
        if sketchlib_out.endswith(".npy"):
            dist_matrix = np.load(os.path.join(data_dir, sketchlib_out))
        elif sketchlib_out.endswith(".pkl"):
            with open(os.path.join(data_dir, sketchlib_out), "rb") as pickle_file:
                try:
                    rlist, qlist, self = pickle.load(pickle_file)
                except ValueError:
                    rlist, qlist = pickle.load(pickle_file)
        else:
            pass

    sources = []
    targets = []
    if self:
        if rlist != qlist:
            raise RuntimeError("rlist must equal qlist for db building (self = true)")
        else:
            for i, ref in enumerate(rlist):
                for j in range(i + 1, len(rlist)):
                    sources.append(rlist[j])
                    targets.append(ref)
    else:
        for query in qlist:
            for ref in rlist:
                sources.append(ref)
                targets.append(query)

    if weights_type == "core":
        core_weights = list(dist_matrix[:,0])
        edge_list = list(zip(sources, targets, core_weights))
    elif weights_type == "accessory":
        acc_weights = list(dist_matrix[:,1])
        edge_list = list(zip(sources, targets, acc_weights))
    elif weights_type == "euclidean":
        euclidean_weights = list(np.linalg.norm(dist_matrix, axis = 1))
        edge_list = list(zip(sources, targets, euclidean_weights))
    else:
        raise RuntimeError("Invalid weight type specified. Valid types are 'core', 'accessory', or 'euclidean'.")

    ref_list = set(rlist).union(set(qlist))

    network_instance = Network(ref_list=ref_list, outdir=outdir, backend=backend)
    network_instance.construct(edge_list, True)

    return network_instance