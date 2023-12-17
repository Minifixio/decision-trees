from typing import List
import numpy as np

from PointSet import PointSet, FeaturesTypes

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        h : int
            The maximum height of the tree
        feature_id : int
            The id of the feature used to split the tree
        feature_threshold : float
            The threshold used to split the tree. Is set to None if the feature is not continous
        feature_split : Tuple[List[float], List[float]]
            The split used to split the tree. Is set to None if the feature is continous
        left : Tree
            The left child of the tree
        right : Tree
            The right child of the tree
        min_split_points : int
            The minimum number of points required to split a node
    """
    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1,
                 min_split_points: int = 1,
                 beta : float = 0):
        """
        Parameters
        ----------
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            labels : List[bool]
                The labels of the training points.
            types : List[FeaturesTypes]
                The types of the features.
            h : int
                The maximum height of the tree.
            min_split_points : int
                The minimum number of points required to split a node.
        """

        self.h = h
        self.feature_id = None
        self.feature_threshold = None
        self.feature_split = None
        self.points = PointSet(features, labels, types)
        self.min_split_points = min_split_points
        self.counter = 0
        self.beta = beta

        if h > 0:
            self.build_tree()


    def build_tree(self, debug=False):
        self.counter = 0            

        if self.h > 0:    
            (feature_id, best_gain) = self.points.get_best_gain(min_split_points=self.min_split_points)
            
            if feature_id == None:
                if (debug): print("->> Tree : Single feature")
                self.h = 0
                return

            is_continous = (self.points.types[feature_id] == FeaturesTypes.REAL)
            threshold = self.points.get_best_threshold()
            
            if not is_continous:
                if threshold != None:
                    split = ([threshold], list(set([f for f in self.points.features[:,feature_id] if f != threshold])))
                else:
                    split = ([0], [1])
                self.feature_split = split
            else:
                self.feature_threshold = threshold

            self.feature_id = feature_id
            
            left_features, left_labels = [], []
            right_features, right_labels = [], []

            for i, point in enumerate(self.points.features):
                if (is_continous and point[self.feature_id] < threshold) or (not is_continous and point[self.feature_id] in split[0]):
                    left_features.append(point)
                    left_labels.append(self.points.labels[i])
                else:
                    right_features.append(point)
                    right_labels.append(self.points.labels[i])
                
            
            if (debug):
                print("->> Tree :")
                print("     spliting on a continous feature ? " + ("YES" if is_continous else "NO"))
                print("     feature id : ", self.feature_id)
                print("     split : ", self.feature_split)
                print("     threshold : ", self.feature_threshold)
                print("     left_features", len(left_features))
                print("     right_features", len(right_features))
                print("\n")

            left_points_count = len(left_features)
            right_points_count = len(right_features)

            if left_points_count < self.min_split_points or right_points_count < self.min_split_points:
                if (debug): print("->> Tree : Left points count or right points count < min_split_points")
                self.h = 0
                return

            self.left = Tree(left_features, left_labels, self.points.types, self.h - 1, self.min_split_points, self.beta)
            self.right = Tree(right_features, right_labels, self.points.types, self.h - 1, self.min_split_points, self.beta)
                
    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        if self.h == 0:
            true_count = np.count_nonzero(self.points.labels)
            false_count = len(self.points.labels) - true_count
            return true_count >= false_count
    
        else:
            if self.feature_split == None:
                if features[self.feature_id] < self.feature_threshold:
                    return self.left.decide(features)
                else:
                    return self.right.decide(features)
            else:
                if features[self.feature_id] in self.feature_split[0]:
                    return self.left.decide(features)
                else:
                    return self.right.decide(features)
    
    def add_training_point(self, features: List[float], label: bool):
        self.counter += 1
        
        if self.h > 0:
            self.points = PointSet(self.points.features.tolist() + [features], self.points.labels.tolist() + [label], self.points.types)

        if self.counter >= self.beta * (len(self.points.features)+0):
            self.counter = 0
            indices_to_remove = []

            for i, point in enumerate(self.points.features):
                if all(f == point[j] for j, f in enumerate(features)):
                    indices_to_remove.append(i)

            self.build_tree()
            return
            
        if self.h == 0:
            return    
        else:
            if self.feature_split == None:
                if features[self.feature_id] < self.feature_threshold:
                    self.left.add_training_point(features, label)
                else:
                    self.right.add_training_point(features, label)
            else:
                if features[self.feature_id] in self.feature_split[0]:
                    self.left.add_training_point(features, label)
                else:
                    self.right.add_training_point(features, label)

    def del_training_point(self, features: List[float], label: bool):
        self.counter += 1
        indices_to_remove = []

        for i, point in enumerate(self.points.features):
            if all(f == point[j] for j, f in enumerate(features)):
                indices_to_remove.append(i)
                break

        if self.h > 0:
            self.points = PointSet([point for i, point in enumerate(self.points.features) if i not in indices_to_remove], [l for i, l in enumerate(self.points.labels) if i not in indices_to_remove], self.points.types)
        
        if self.counter >= self.beta * (len(self.points.features)):
            self.counter = 0
            self.build_tree()
            return
        
        if self.h == 0:
            return
        else:
            if self.feature_split == None:
                if features[self.feature_id] < self.feature_threshold:
                    self.left.del_training_point(features, label)
                else:
                    self.right.del_training_point(features, label)
            else:
                if features[self.feature_id] in self.feature_split[0]:
                    self.left.del_training_point(features, label)
                else:
                    self.right.del_training_point(features, label)

    def print_tree(self, tree, depth=0):
        print("--" * depth, end="")

        if tree.h == 0:
            true_count = np.count_nonzero(tree.points.labels)
            false_count = len(tree.points.labels) - true_count
            print("Leaf: Decision = " + str(true_count >= false_count), " | true:" + str(true_count) + ", false:" + str(false_count))
        else:
            print(f"Node: Feature {tree.feature_id}, Split {tree.feature_split if tree.feature_split is not None else tree.feature_threshold}")

        if hasattr(tree, "left") and tree.left:
            self.print_tree(tree.left, depth + 1)
        if hasattr(tree, "right") and tree.right:
            self.print_tree(tree.right, depth + 1)