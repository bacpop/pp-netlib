import os, sys
from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
import subprocess

import h5py
#import rmm
#import cudf
#import cupy

from numba import cuda


try:
    import cupyx
    import cugraph
    import cudf
    import cupy as cp
    from numba import cuda
    import rmm
    gpu_lib = True
except ImportError as e:
    gpu_lib = False


betweenness_sample_default = 100
accepted_weights_types = ["euclidean", "core", "accesory"]

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="PopPUNK network utilities")

    parser.add_argument('--ref-db',type = str, help='Location of built reference database')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads to use [default = 1]')
    parser.add_argument("--graph_weights", help="Save within-strain Euclidean distances into the graph", default=False, action="store_true")
    parser.add_argument("--distances", help="Prefix of input pickle of pre-calculated distances")
    parser.add_argument("--ranks", help="Comma separated list of ranks used in lineage clustering [default = 1,2,3]", type = str, default = "1,2,3")
    parser.add_argument('--betweenness_sample', help='Number of sequences used to estimate betweeness with a GPU [default = 100]', type = int, default = betweenness_sample_default)
    parser.add_argument("--gpu_graph", default=False, action="store_true", help="Use a GPU when calculating networks [default = False]")
    parser.add_argument("--external_clustering", help="File with cluster definitions or other labels generated with any other method.", default=None)
    parser.add_argument("--indiv_refine", help="Also run refinement for core and accessory individually", choices=["both", "core", "accessory"], default=None)
    parser.add_argument("--output", help="Prefix for output files")

    args = parser.parse_args()
    return args

def read_isolate_type_from_csv(clusters_csv, mode = "clusters", return_dict = False):
    """Read cluster definitions from CSV file.
    Args:
        clusters_csv (str)
            File name of CSV with isolate assignments
        return_type (str)
            If True, return a dict with sample->cluster instead
            of sets
    Returns:
        clusters (dict)
            Dictionary of cluster assignments (keys are cluster names, values are
            sets containing samples in the cluster). Or if return_dict is set keys
            are sample names, values are cluster assignments.
    """
    # data structures
    if return_dict:
        clusters = defaultdict(dict)
    else:
        clusters = {}

    # read CSV
    clusters_csv_df = pd.read_csv(clusters_csv, index_col = 0, quotechar='"')

    # select relevant columns according to mode
    if mode == "clusters":
        type_columns = [n for n,col in enumerate(clusters_csv_df.columns) if ("Cluster" in col)]
    elif mode == "lineages":
        type_columns = [n for n,col in enumerate(clusters_csv_df.columns) if ("Rank_" in col or "overall" in col)]
    elif mode == "external":
        if len(clusters_csv_df.columns) == 1:
            type_columns = [0]
        elif len(clusters_csv_df.columns) > 1:
            type_columns = range((len(clusters_csv_df.columns)-1))
    else:
        sys.stderr.write("Unknown CSV reading mode: " + mode + "\n")
        sys.exit(1)

    # read file
    for row in clusters_csv_df.itertuples():
        for cls_idx in type_columns:
            cluster_name = clusters_csv_df.columns[cls_idx]
            cluster_name = cluster_name.replace("__autocolour","")
            if return_dict:
                clusters[cluster_name][str(row.Index)] = str(row[cls_idx + 1])
            else:
                if cluster_name not in clusters.keys():
                    clusters[cluster_name] = defaultdict(set)
                clusters[cluster_name][str(row[cls_idx + 1])].add(row.Index)

    # return data structure
    return clusters

def gen_unword(unique=True):
    import json
    import gzip
    import random
    import string
    import pkg_resources

    # Download from https://github.com/dwyl/english-words/raw/master/words_dictionary.json
    word_list = pkg_resources.resource_stream(__name__, "data/words_dictionary.json.gz")
    with gzip.open(word_list, "rb") as word_list:
        real_words = json.load(word_list)

    vowels = ["a", "e", "i", "o", "u"]
    trouble = ["q", "x", "y"]
    consonants = set(string.ascii_lowercase) - set(vowels) - set(trouble)

    vowel = lambda: random.sample(vowels, 1)
    consonant = lambda: random.sample(consonants, 1)
    cv = lambda: consonant() + vowel()
    cvc = lambda: cv() + consonant()
    syllable = lambda: random.sample([vowel, cv, cvc], 1)

    returned_words = set()
    # Iterator loop
    while True:
        # Retry loop
        while True:
            word = ""
            for i in range(random.randint(2, 3)):
                word += "".join(syllable()[0]())
            if word not in real_words and (not unique or word not in returned_words):
                returned_words.add(word)
                break
        yield word

def check_and_set_gpu(use_gpu, gpu_lib, quit_on_fail = False):
    """Check GPU libraries can be loaded and set managed memory.
    Args:
        use_gpu (bool)
            Whether GPU packages have been requested
        gpu_lib (bool)
            Whether GPU packages are available
    Returns:
        use_gpu (bool)
            Whether GPU packages can be used
    """
    # load CUDA libraries
    if use_gpu and not gpu_lib:
        if quit_on_fail:
            sys.stderr.write("Unable to load GPU libraries; exiting\n")
            sys.exit(1)
        else:
            sys.stderr.write("Unable to load GPU libraries; using CPU libraries "
            "instead\n")
            use_gpu = False

    # Set memory management for large networks
    if use_gpu:
        rmm.reinitialize(managed_memory=True)
        cudf.set_allocator("managed")
        if "cupy" in sys.modules:
            cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)
        if "cuda" in sys.modules:
            cuda.set_memory_manager(rmm.RMMNumbaManager)
        assert(rmm.is_initialized())

    return use_gpu

def generate_tuples():
    #TODO - from poppunk-refine
    return

def generate_all_tuples():
    #TODO - from poppunk_refine
    return

def add_random(out_prefix, sequence_names, klist, strand_preserved = False, overwrite = False, threads = 1):
    """Add chance of random match to a HDF5 sketch DB
    Args:
        out_prefix (str)
            Sketch database prefix
        sequence_names (list)
            Names of sequences to include in calculation
        klist (list)
            List of k-mer sizes to sketch
        strand_preserved (bool)
            Set true to ignore rc k-mers
        overwrite (str)
            Set true to overwrite existing random match chances
        threads (int)
            Number of threads to use (default = 1)
    """
    if len(sequence_names) <= 2:
        sys.stderr.write("Cannot add random match chances with this few genomes\n")
    else:
        dbname = out_prefix + "/" + os.path.basename(out_prefix)
        hdf_in = h5py.File(dbname + ".h5", "r+")

        if "random" in hdf_in:
            if overwrite:
                del hdf_in['random']
            else:
                sys.stderr.write("Using existing random match chances in DB\n")
                return

        hdf_in.close()
        add_random(dbname, sequence_names, klist, not strand_preserved, threads)

def iter_dist_rows(ref_seqs, query_seqs, self=True):
    """Gets the ref and query ID for each row of the distance matrix
    Returns an iterable with ref and query ID pairs by row.
    Args:
        ref_seqs (list)
            List of reference sequence names.
        query_seqs (list)
            List of query sequence names.
        self (bool)
            Whether a self-comparison, used when constructing a database.
            Requires ref_seqs == querySeqs
            Default is True
    Returns:
        ref, query (str, str)
            Iterable of tuples with ref and query names for each distMat row.
    """
    if self:
        if ref_seqs != query_seqs:
            raise RuntimeError("ref_seqs must equal querySeqs for db building (self = true)")
        for i, ref in enumerate(ref_seqs):
            for j in range(i + 1, len(ref_seqs)):
                yield(ref_seqs[j], ref)
    else:
        for query in query_seqs:
            for ref in ref_seqs:
                yield(ref, query)

def read_pickle(pkl_filename, enforce_self=False, distances=True):
    """Loads core and accessory distances saved by :func:`~storePickle`
    Called during ``--fit-model``
    Args:
        pkl_filename (str)
            Prefix for saved files
        enforce_self (bool)
            Error if self == False
            [default = True]
        distances (bool)
            Read the distance matrix
            [default = True]
    Returns:
        rlist (list)
            List of reference sequence names (for :func:`~iterDistRows`)
        qlist (list)
            List of query sequence names (for :func:`~iterDistRows`)
        self (bool)
            Whether an all-vs-all self DB (for :func:`~iterDistRows`)
        X (numpy.array)
            n x 2 array of core and accessory distances
    """
    with open(pkl_filename + ".pkl", 'rb') as pickle_file:
        rlist, qlist, self = pickle.load(pickle_file)
        if enforce_self and not self:
            sys.stderr.write("Old distances " + pkl_filename + ".npy not complete\n")
            sys.stderr.exit(1)
    if distances:
        X = np.load(pkl_filename + ".npy")
    else:
        X = None
    return rlist, qlist, self, X

def store_pickle(rlist, qlist, self, X, pklName):
    """Saves core and accessory distances in a .npy file, names in a .pkl
    Called during ``--create-db``
    Args:
        rlist (list)
            List of reference sequence names (for :func:`~iterDistRows`)
        qlist (list)
            List of query sequence names (for :func:`~iterDistRows`)
        self (bool)
            Whether an all-vs-all self DB (for :func:`~iterDistRows`)
        X (numpy.array)
            n x 2 array of core and accessory distances
        pklName (str)
            Prefix for output files
    """
    with open(pklName + ".pkl", 'wb') as pickle_file:
        pickle.dump([rlist, qlist, self], pickle_file)
    np.save(pklName + ".npy", X)

def create_overall_lineage(rank_list, lineage_clusters):
    # process multirank lineages
    overall_lineages = {"Rank_" + str(rank):{} for rank in rank_list}
    overall_lineages["overall"] = {}
    isolate_list = lineage_clusters[rank_list[0]].keys()
    for isolate in isolate_list:
        overall_lineage = None
        for rank in rank_list:
            overall_lineages["Rank_" + str(rank)][isolate] = lineage_clusters[rank][isolate]
            if overall_lineage is None:
                overall_lineage = str(lineage_clusters[rank][isolate])
            else:
                overall_lineage = overall_lineage + '-' + str(lineage_clusters[rank][isolate])
        overall_lineages["overall"][isolate] = overall_lineage

    return overall_lineages

def isolate_name_to_label(names):
    """Function to process isolate names to labels
    appropriate for visualisation.
    Args:
        names (list)
            List of isolate names.
    Returns:
        labels (list)
            List of isolate labels.
    """
    # useful to have as a function in case we
    # want to remove certain characters
    labels = [name.split("/")[-1].replace(".","_").replace(":","").replace("(","_").replace(")","_") for name in names]
    
    return labels

def write_cluster_csv(outfile, nodeNames, nodeLabels, clustering, output_format = "microreact", epiCsv = None, queryNames = None, suffix = "_Cluster"):
    """Print CSV file of clustering and optionally epi data
    Writes CSV output of clusters which can be used as input to microreact and cytoscape.
    Uses pandas to deal with CSV reading and writing nicely.
    The epiCsv, if provided, should have the node labels in the first column.
    Args:
        outfile (str)
            File to write the CSV to.
        nodeNames (list)
            Names of sequences in clustering (includes path).
        nodeLabels (list)
            Names of sequences to write in CSV (usually has path removed).
        clustering (dict or dict of dicts)
            Dictionary of cluster assignments (keys are nodeNames). Pass a dict with depth two
            to include multiple possible clusterings.
        output_format (str)
            Software for which CSV should be formatted
            (microreact, phandango, grapetree and cytoscape are accepted)
        epiCsv (str)
            Optional CSV of epi data to paste in the output in addition to
            the clusters (default = None).
        queryNames (list)
            Optional list of isolates that have been added as a query.
            (default = None)
    """
    # set order of column names
    colnames = []
    if output_format == "microreact":
        colnames = ["id"]
        for cluster_type in clustering:
            col_name = cluster_type + suffix + "__autocolour"
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append("Status")
            colnames.append("Status__colour")
    elif output_format == "phandango":
        colnames = ['id']
        for cluster_type in clustering:
            col_name = cluster_type + suffix
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append("Status")
            colnames.append("Status:colour")
    elif output_format == "grapetree":
        colnames = ["ID"]
        for cluster_type in clustering:
            col_name = cluster_type + suffix
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append("Status")
    elif output_format == "cytoscape":
        colnames = ["id"]
        for cluster_type in clustering:
            col_name = cluster_type + suffix
            colnames.append(col_name)
        if queryNames is not None:
            colnames.append("Status")
    else:
        sys.stderr.write("Do not recognise format for CSV writing\n")
        exit(1)

    # process epidemiological data
    d = defaultdict(list)

    # process epidemiological data without duplicating names
    # used by PopPUNK
    columns_to_be_omitted = ["id", "Id", "ID", "combined_Cluster__autocolour",
    "core_Cluster__autocolour", "accessory_Cluster__autocolour",
    "overall_Lineage"]
    if epiCsv is not None:
        epiData = pd.read_csv(epiCsv, index_col = False, quotechar='"')
        epiData.index = isolate_name_to_label(epiData.iloc[:,0])
        for e in epiData.columns.values:
            if e not in columns_to_be_omitted:
                colnames.append(str(e))

    # get example clustering name for validation
    example_cluster_title = list(clustering.keys())[0]

    for name, label in zip(nodeNames, isolate_name_to_label(nodeLabels)):
        if name in clustering[example_cluster_title]:
            if output_format == "microreact":
                d["id"].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix + "__autocolour"
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d["Status"].append("Query")
                        d["Status__colour"].append("red")
                    else:
                        d["Status"].append("Reference")
                        d["Status__colour"].append("black")
            elif output_format == 'phandango':
                d["id"].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d["Status"].append("Query")
                        d["Status:colour"].append("#ff0000")
                    else:
                        d["Status"].append("Reference")
                        d["Status:colour"].append("#000000")
            elif output_format == "grapetree":
                d["ID"].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d["Status"].append("Query")
                    else:
                        d["Status"].append("Reference")
            elif output_format == "cytoscape":
                d["id"].append(label)
                for cluster_type in clustering:
                    col_name = cluster_type + suffix
                    d[col_name].append(clustering[cluster_type][name])
                if queryNames is not None:
                    if name in queryNames:
                        d["Status"].append("Query")
                    else:
                        d["Status"].append("Reference")
            if epiCsv is not None:
                if label in epiData.index:
                    for col, value in zip(epiData.columns.values, epiData.loc[label].values):
                        if col not in columns_to_be_omitted:
                            d[col].append(str(value))
                else:
                    for col in epiData.columns.values:
                        if col not in columns_to_be_omitted:
                            d[col].append("nan")

        else:
            sys.stderr.write("Cannot find " + name + " in clustering\n")
            sys.exit(1)

    # print CSV
    sys.stderr.write("Parsed data, now writing to CSV\n")
    try:
        pd.DataFrame(data=d).to_csv(outfile, columns = colnames, index = False)
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Problem with epidemiological data CSV; returned code: " + str(e.returncode) + "\n")
        # check CSV
        prev_col_items = -1
        prev_col_name = "unknown"
        for col in d:
            this_col_items = len(d[col])
            if prev_col_items > -1 and prev_col_items != this_col_items:
                sys.stderr.write("Discrepant length between " + prev_col_name + \
                                 " (length of " + prev_col_items + ") and " + \
                                 col + "(length of " + this_col_items + ")\n")
        sys.exit(1)

def prune_distance_matrix(refList, remove_seqs_in, dist_matrix, output):
    """Rebuild distance matrix following selection of panel of references
    Args:
        refList (list)
            List of sequences used to generate distance matrix
        remove_seqs_in (list)
            List of sequences to be omitted
        dist_matrix (numpy.array)
            nx2 matrix of core distances (column 0) and accessory
            distances (column 1)
        output (string)
            Prefix for new distance output files
    Returns:
        newRefList (list)
            List of sequences retained in distance matrix
        newDistMat (numpy.array)
            Updated version of distMat
    """
    # Find list items to remove
    remove_seqs_list = []
    removal_indices = []
    for to_remove in remove_seqs_in:
        found = False
        for idx, item in enumerate(refList):
            if item == to_remove:
                removal_indices.append(idx)
                remove_seqs_list.append(item)
                found = True
                break
        if not found:
            sys.stderr.write("Couldn't find " + to_remove + " in database\n")
    remove_seqs = frozenset(remove_seqs_list)

    if len(remove_seqs) > 0:
        sys.stderr.write("Removing " + str(len(remove_seqs)) + " sequences\n")

        num_new = len(refList) - len(remove_seqs)
        new_dist_matrix = np.zeros((int(0.5 * num_new * (num_new - 1)), 2), dtype=dist_matrix.dtype)

        # Create new reference list iterator
        removal_indices.sort()
        removal_indices.reverse()
        next_remove = removal_indices.pop()
        new_ref_list = []
        for idx, seq in enumerate(refList):
            if idx == next_remove:
                if len(removal_indices) > 0:
                    next_remove = removal_indices.pop()
            else:
                new_ref_list.append(seq)

        new_row_names = iter(iter_dist_rows(new_ref_list, new_ref_list, self=True))

        # Copy over rows which don't have an excluded sequence
        new_idx = 0
        for dist_row, (ref1, ref2) in zip(dist_matrix, iter_dist_rows(refList, refList, self=True)):
            if ref1 not in remove_seqs and ref2 not in remove_seqs:
                (newRef1, newRef2) = next(new_row_names)
                if newRef1 == ref1 and newRef2 == ref2:
                    new_dist_matrix[new_idx, :] = dist_row
                    new_idx += 1
                else:
                    raise RuntimeError("Row name mismatch. Old: " + ref1 + "," + ref2 + "\n"
                                       "New: " + newRef1 + "," + newRef2 + "\n")

        store_pickle(new_ref_list, new_ref_list, True, new_dist_matrix, output)
    else:
        new_ref_list = refList
        new_dist_matrix = dist_matrix

    # return new distance matrix and sequence lists
    return new_ref_list, new_dist_matrix

def remove_from_db(db_name, out_name, remove_seqs, full_names = False):
    """Remove sketches from the DB the low-level HDF5 copy interface
    Args:
        db_name (str)
            Prefix for hdf database
        out_name (str)
            Prefix for output (pruned) database
        remove_seqs (list)
            Names of sequences to remove from database
        full_names (bool)
            If True, db_name and out_name are the full paths to h5 files
    """
    remove_seqs = set(remove_seqs)
    if not full_names:
        db_file = db_name + "/" + os.path.basename(db_name) + ".h5"
        out_file = out_name + "/" + os.path.basename(out_name) + ".tmp.h5"
    else:
        db_file = db_name
        out_file = out_name

    hdf_in = h5py.File(db_file, "r")
    hdf_out = h5py.File(out_file, "w")

    try:
        if "random" in hdf_in.keys():
            hdf_in.copy("random", hdf_out)
        out_grp = hdf_out.create_group("sketches")
        read_grp = hdf_in["sketches"]
        for attr_name, attr_val in read_grp.attrs.items():
            out_grp.attrs.create(attr_name, attr_val)

        removed = []
        for dataset in read_grp:
            if dataset not in remove_seqs:
                out_grp.copy(read_grp[dataset], dataset)
            else:
                removed.append(dataset)
    except RuntimeError as e:
        sys.stderr.write("ERROR: " + str(e) + "\n")
        sys.stderr.write("Error while deleting sequence " + dataset + "\n")
        sys.exit(1)

    missed = remove_seqs.difference(set(removed))
    if len(missed) > 0:
        sys.stderr.write("WARNING: Did not find samples to remove:\n")
        sys.stderr.write("\t".join(missed) + "\n")

    # Clean up
    hdf_in.close()
    hdf_out.close()

