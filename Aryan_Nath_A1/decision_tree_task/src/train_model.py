import numpy as np
import pandas as pd

class Node:
    def __init__(self, datapoints):
        self.datapoints = datapoints
        self.feature_index = None
        self.threshold = None
        self.right = None
        self.left = None
        self.is_leafnode = False
        self.depth = 0

class DecisionTree:
    def __init__(self, max_depth, min_size, datapoints):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = Node(datapoints)

    def get_entropy(self, node):
        X = node.datapoints
        # for binary classification
        class_0 = X[X.iloc[:,-1] == 0]
        class_1 = X[X.iloc[:,-1] == 1]
        if len(class_0) == 0 or len(class_1) == 0:
            return 0
        p_0 = len(class_0) / len(X)
        p_1 = len(class_1) / len(X)
        entropy = -1 * (p_0 * np.log2(p_0) + p_1 * np.log2(p_1))
        return entropy
    
    def split_data(self, node, feature_index, threshold):
        X = node.datapoints
        left_split = X[X.iloc[:, feature_index] <= threshold]
        right_split = X[X.iloc[:, feature_index] > threshold]
        return left_split, right_split
    
    def find_best_split(self, node):
        X = node.datapoints
        entropy_parent = self.get_entropy(node)
        
        best_information_gain = -float('inf')
        best_feature_index = None
        best_feature_val = None

        for feature_index in range(X.shape[1] - 1):
            feature_values = X.iloc[:, feature_index].unique()
            for feature_val in feature_values:
                left_split, right_split = self.split_data(node, feature_index, feature_val)
                
                if len(left_split) == 0 or len(right_split) == 0:
                    continue
                
                left_entropy = self.get_entropy(Node(left_split))
                right_entropy = self.get_entropy(Node(right_split))
                
                # get the information gain
                information_gain = entropy_parent - (len(left_split) / len(X) * left_entropy + len(right_split) / len(X) * right_entropy)
                
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_feature_index = feature_index
                    best_feature_val = feature_val

        return best_feature_index, best_feature_val, best_information_gain

    def build_tree(self, node, depth):
        print(f"current node at depth: {depth}, datapoints left: {len(node.datapoints)}, unique classes: {set(node.datapoints.iloc[:,-1])}")
        if len(node.datapoints) < self.min_size:
            node.is_leafnode = True
            print(f"leaf node created at depth {depth} because of the min points conditions")
            print("entropy:",self.get_entropy(node))
            return node
        
        if depth >= self.max_depth:
            node.is_leafnode = True
            print(f"leaf node created at depth {depth} because of the max depth condition")
            print("entropy:",self.get_entropy(node))
            return node
        
        # impurity of the current node is 0
        if self.get_entropy(node) == 0:
            node.is_leafnode = True
            print(f"leaf node created at depth {depth} as the node has 0 impurity")
            return node
        
        best_feature_index, best_feature_val, information_gain = self.find_best_split(node)
        
        if information_gain <= 0:
            node.is_leafnode = True
            print(f"leaf node created at depth {depth} as there is no information gain")
            return node
        
        left_split, right_split = self.split_data(node, best_feature_index, best_feature_val)
        
        if len(left_split) == 0 or len(right_split) == 0:
            node.is_leafnode = True
            print(f"leaf node created at depth {depth} as the split does not create two children")
            return node
        
        node.feature_index = best_feature_index
        node.threshold = best_feature_val
        print(f"node created at depth {depth} with feature={features[best_feature_index]}, threshold={best_feature_val} and entropy={self.get_entropy(node)}")
        
        node.left = self.build_tree(Node(left_split), depth + 1)
        node.right = self.build_tree(Node(right_split), depth + 1)
        return node
    
    def fit(self, X):
        print("fitting the decision tree...")
        self.root = self.build_tree(self.root, 0)
        print("fitting complete")

    def pred_node(self, node):
        pred_dict = {1: 0, 0: 0}
        for index, datapoint in node.datapoints.iterrows():
            pred_dict[datapoint.iloc[-1]] += 1
        return max(pred_dict, key=pred_dict.get)

    def predict(self,X):
        num_datapoints = X.shape[0]
        predictions = []
        for i in range(num_datapoints):
            datapoint = X.iloc[i]
            node_cur = self.root
            while not node_cur.is_leafnode:
                if datapoint[node_cur.feature_index] <= node_cur.threshold:
                    node_cur = node_cur.left
                else:
                    node_cur = node_cur.right
            predictions.append(self.pred_node(node_cur))
        return predictions

# Data preparation
data_directory = '../data/'
df = pd.read_csv(data_directory + 'fraud_train.csv')
X = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].copy()
X['diffOrig'] = X['newbalanceOrig'] - X['oldbalanceOrg']
X['diffDest'] = X['newbalanceDest'] - X['oldbalanceDest']
features = X.columns
X['isFraud'] = df['isFraud']

dt = DecisionTree(max_depth=20, min_size=1, datapoints=X.head(10000))
dt.fit(dt.root.datapoints)

predictions = dt.predict(X.head(10000))

print(predictions.count(1))
print(predictions.count(0))
print(X.head(10000).iloc[:,-1].value_counts())

matches = [True for i in range(len(predictions)) if predictions[i] == X.head(10000).iloc[i,-1]]