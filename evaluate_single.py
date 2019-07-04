import os
import json
import multiprocessing as mp
from functools import partial
from algorithms import RegionGrowthSimple, RegionGrowthExponent, RegionGrowthVar, FastScanning, Felzenszwalb, Slic, Watershed
### Evaluate Single results
import evaluation
from algorithms import MaskRCNN

ROOT_DIR = os.path.abspath("")
results_dir = os.path.join(ROOT_DIR, "results")


def evaluate_var(config, data):
    alg = RegionGrowthVar(data[0])
    return alg.evaluate(config, data[1])


def evaluate_exponent(config, data):
    alg = RegionGrowthExponent(data[0])
    return alg.evaluate(config, data[1])


def evaluate_simple(config, data):
    alg = RegionGrowthSimple(data[0])
    return alg.evaluate(config, data[1])


def evaluate_fast_scanning(config, data):
    alg = FastScanning(data[0])
    return alg.evaluate(config, data[1])


def evaluate_slic(config, data):
    alg = Slic(data[0])
    return alg.evaluate(config, data[1])


def evaluate_watershed(config, data):
    alg = Watershed(data[0])
    return alg.evaluate(config, data[1])


def evaluate_felzenszwalb(config, data):
    alg = Felzenszwalb(data[0])
    return alg.evaluate(config, data[1])


def evaluate_MaskRCNN(data):
    alg = MaskRCNN()
    return alg.evaluate(data[0], data[1])


def evaluate(alg_name, data, config):
    # Parallelizing using Pool.apply()
    num_cpus = mp.cpu_count()

    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(num_cpus)

    # Step 2: `pool.apply` the `howmany_within_range()`
    if alg_name == 'region_growth_var':
        results = pool.map(partial(evaluate_var, config), data)
    if alg_name == 'region_growth_exponent':
        results = pool.map(partial(evaluate_exponent, config), data)
    if alg_name == 'region_growth_simple':
        results = pool.map(partial(evaluate_simple, config), data)
    if alg_name == 'watershed':
        results = pool.map(partial(evaluate_watershed, config), data)
    if alg_name == 'slic':
        results = pool.map(partial(evaluate_slic, config), data)
    if alg_name == 'felzenszwalb':
        results = pool.map(partial(evaluate_felzenszwalb, config), data)
    if alg_name == 'fast_scanning':
        results = pool.map(partial(evaluate_fast_scanning, config), data)
    if alg_name == 'mask-rcnn':
        results = pool.map(evaluate_MaskRCNN, data)

    # Step 3: Don't forget to close
    pool.close()

    return results

def avg_single_results(results):
    npri_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f_score_sum = 0.0

    num_images = len(results)

    for configurations in results:
        npri_sum += configurations["npri_score"]
        precision_sum += configurations["precision"]
        recall_sum += configurations["recall"]
        f_score_sum += configurations["f_score"]

    results[0]["npri_score"] = (npri_sum / num_images)
    results[0]["precision"] = (precision_sum / num_images)
    results[0]["recall"] = (recall_sum / num_images)
    results[0]["f_score"] = (f_score_sum / num_images)

    return results[0]


def print_results(results):
    print("rand-index: {}".format(results['npri_score']))
    print("f_score: {}".format(results['f_score']))
    print("precision: {}".format(results['precision']))
    print("recall: {}".format(results['recall']))


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run single evaluation')
    parser.add_argument("algorithm",
                        metavar="<algorithm_name>",
                        help="Name of the algorithm e.g. felzenszwalb")

    parser.add_argument('config',
                        metavar="<XXXX>",
                        help='JSON string for of config values')

    parser.add_argument('--colorspace',
                        metavar="<colorspace_name>",
                        default='RGB',
                        help='Name of the colorspace to convert to e.g. BGR, YCrCb')
    parser.add_argument('--dataset',
                        metavar="evaluation | training",
                        default='training',
                        help='Whether to use evaluation or training dataset')
    parser.add_argument('--width',
                        metavar="<XXXX>",
                        default=512,
                        help='Image width for downscaling')

    args = parser.parse_args()

    algorithm_name = args.algorithm
    cspace = args.colorspace
    config = json.loads(args.config)

    data_pairs = evaluation.get_data(evaluation.colorspace2function[cspace], width=args.width, dataset=args.dataset)
    results = evaluate(algorithm_name, data_pairs, config)
    result = avg_single_results(results)
    print_results(result)
