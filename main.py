from PointSet import PointSet
from Tree import Tree
from read_write import load_data
import evaluation
import argparse
    
def build(file_to_study, tree_size_proportion=.8, **tree_params):
    if tree_size_proportion is None: tree_size_proportion = .8
    features, labels, types = load_data(file_to_study)
    training_nb = int(len(features)*tree_size_proportion)
    current_tree = Tree(features[:training_nb], labels[:training_nb], types, **tree_params)
    expected_results = labels[training_nb:]
    actual_results = []
    for point_features in features[training_nb:]:
        actual_results += [current_tree.decide(point_features)]
    current_tree.print_tree(current_tree)
    results = evaluation.F1_score(expected_results, actual_results)
    return results


def build_FuDyADT(file_to_study, tree_size_proportion=.3, **tree_params):
    if tree_size_proportion is None: tree_size_proportion = .3
    features, labels, types = load_data(file_to_study)
    training_nb = int(len(features)*tree_size_proportion)
    current_tree = Tree(features[:training_nb], labels[:training_nb], types, **tree_params)
    expected_results = labels[training_nb:]
    actual_results = []
    for i, (point_features, point_label)\
            in enumerate(zip(features[training_nb:], labels[training_nb:])):
        actual_results += [current_tree.decide(point_features)]
        current_tree.add_training_point(point_features, point_label)
        del_point_feat, del_point_lab = features[i], labels[i]
        current_tree.del_training_point(del_point_feat, del_point_lab)
    results = evaluation.F1_score(expected_results, actual_results)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command-line tool with optional arguments")
    parser.add_argument("file_path", type=str, help="Path to the file")

    parser.add_argument("-hgt", "--height", type=int, help="Height (optional)")
    parser.add_argument("-msp", "--min_split_points", type=int, help="Min split points (optional)")
    parser.add_argument("-tsp", "--tree_size_proportion", type=int, help="Tree size proportion (optional)")
    parser.add_argument("-fudyadt", action="store_true", help="Use the FuDyADT method (optional)")

    args = parser.parse_args()

    file_path = args.file_path
    height = args.height
    if height is None: height = 5
    min_split_points = args.min_split_points
    if min_split_points is None: min_split_points = 3
    tree_size_proportion = args.tree_size_proportion
    fudyadt = args.fudyadt
    
    if not fudyadt:
        r = build(file_path, tree_size_proportion=tree_size_proportion, h=height, min_split_points=min_split_points)
        print("f1 score : " + str(r))
    else:
        r = build_FuDyADT(file_path, tree_size_proportion=tree_size_proportion, h=height, min_split_points=min_split_points)
        print("f1 score : " + str(r))