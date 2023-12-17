from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = [ t for t in types if t is not None ] # remove None types to avoid errors
        self.features = np.array(features)
        self.labels = np.array(labels)
        self.split = None
        self.threshold = None
        self.selected_feature_id = None
        self.category_id = None
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """

        total = self.labels.size
        count_label_true = 0

        for label in self.labels:
            if label:
                count_label_true += 1
        
        return 1 - (count_label_true/total)**2 - ((total-count_label_true)/total)**2


    def get_best_gain(self, min_split_points: int=None) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain
        Parameters
        ----------
        min_split_points : int
            The minimum number of points in each split to consider the split
            Default is None, which means that there is no minimum number of points in each split

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """

        if len(self.features) == 0:
            return (None, None)

        total = self.labels.size
        res = (None, None)
        threshold_found = None
        category_id_found = None

        right_count, left_count = 0, 0 # will be useful to check if the split is relevant according to min_split_points

        for i in range(self.features.shape[1]):
                        
            f_values = self.features[:,i] # values of the feature i
            f_type = self.types[i] # type of the feature i

            if f_type == FeaturesTypes.REAL:
                sorted_indices = np.argsort(f_values)
                sorted_labels = np.array(self.labels)[sorted_indices]
                values = np.array(f_values)[sorted_indices]
                s = sum(self.labels)
                c00, c01, c10, c11 = 0, 0, total-s, s # the gini coefficients

            elif f_type == FeaturesTypes.CLASSES:
                # since we consider splits like :
                # ([value], [the rest])
                # we only need to consider the values that are in the set
                values = np.unique(f_values)
                
            elif f_type == FeaturesTypes.BOOLEAN:
                values = [0]


            for k, val in enumerate(values):

                # doing the O(nlogn) sort here allows us to avoid doing it for each iteration of the loop
                if f_type == FeaturesTypes.REAL:
                    if k>0 and sorted_labels[k-1] == 1:
                        c01 += 1
                        c11 -= 1
                    elif k>0 and sorted_labels[k-1] == 0:
                        c00 += 1
                        c10 -= 1
                    
                    if k==0 or (k>0 and val == values[k-1]):
                        continue
                else:
                    c00, c01, c10, c11 = 0, 0, 0, 0

                    for j, f in enumerate(f_values):
                        # the following checks work for both boolean and classes features
                        if (not self.labels[j] and f == val): c00 += 1
                        elif (self.labels[j] and f == val): c01 += 1
                        elif (not self.labels[j] and f != val): c10 += 1
                        elif (self.labels[j] and f != val): c11 += 1

                if (c00+c01 == 0 or c10+c11 == 0):
                    continue
            
                gini0 = 1 - (c00/(c00+c01))**2 - (c01/(c00+c01))**2
                gini1 = 1 - (c10/(c10+c11))**2 - (c11/(c10+c11))**2

                gini_split = (c00+c01)/total*gini0 + (c10+c11)/total*gini1
                gini_gain = self.get_gini() - gini_split

                if (res[1] == None or gini_gain > res[1]):
                    right_count = c00 + c01
                    left_count = c10 + c11

                    if min_split_points is not None and (right_count < min_split_points or left_count < min_split_points): 
                        continue

                    if f_type == FeaturesTypes.REAL: threshold_found = val if k == 0 else (values[k-1] + val)/2
                    if f_type == FeaturesTypes.CLASSES: category_id_found = val

                    res = (i, gini_gain)

        
        if res[0] == None: return res

        # if the feature is real, we store the threshold found
        if self.types[res[0]] == FeaturesTypes.REAL: self.threshold = threshold_found 

        # if the feature is a category, we store the category id found
        if self.types[res[0]] == FeaturesTypes.CLASSES: self.category_id = category_id_found 

        # we store the feature id found
        self.selected_feature_id = res[0]

        return res
        
    def get_best_threshold(self) -> float:

        if self.selected_feature_id == None:
            raise Exception('Please call get_best_gain() first')
    
        elif self.types[self.selected_feature_id] == FeaturesTypes.REAL:
            return self.threshold
        
        elif self.types[self.selected_feature_id] == FeaturesTypes.BOOLEAN:
            return None

        elif self.types[self.selected_feature_id] == FeaturesTypes.CLASSES:
            return self.category_id
        
        
