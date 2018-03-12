from typing import Type, Sequence, Dict
import random

class GibbsNode:

    #the actual value of the node is stored as self.value
    #this actual value should be an int, starting at 0
    #probability_dict should take tuples of integers as key, and have a list of probabilities as value
    def __init__(self, node_name : str, value_names : Sequence[str], probability_dict,
                 parents : Sequence['GibbsNode']):
        self.name = node_name
        self.value_names = value_names
        self.p_dict = probability_dict
        self.parents = parents
        self.value = 0 #initialize value as 0 by default
        self.max_value = len(self.value_names)-1
        self.is_fixed = False

    def get_name(self) -> str:
        return self.name

    def set_fixed(self, f : bool):
        self.is_fixed = f

    def get_string_value(self) -> str:
        return self.value_names[self.value]

    def get_current_value(self) -> int:
        return self.value

    def set_value_random(self):
        self.value = random.randint(0, self.max_value)

    def update_node(self):
        if self.is_fixed:
            return
        #get values of parents
        if not self.parents:
            parent_values = ()
        else:
            p_vals = []
            for p in self.parents:
                p_vals.append(p.get_current_value())
            parent_values = tuple(p_vals)
        #get the probability distribution for the current configuration
        probabilities = self.p_dict[parent_values]
        randv = random.random()
        ptotal = 0
        #iterate through and add each probability, use this running total to determine the new value of this node
        for i in range(0, len(probabilities)):
            ptotal += probabilities[i]
            if randv < ptotal:
                self.value = i
                break

def main():
    amenities = GibbsNode("amenities", ["lots", "little"], {(): [0.3, 0.7]}, [])
    neighborhood = GibbsNode("neighborhood", ["bad", "good"], {(): [0.4, 0.6]}, [])
    location = GibbsNode("location", ["good", "bad", "ugly"], {
        (0, 0): [0.3, 0.4, 0.3],
        (0, 1): [0.8, 0.15, 0.05],
        (1, 0): [0.2, 0.4, 0.4],
        (1, 1): [0.5, 0.35, 0.15]
    }, [amenities, neighborhood])
    children = GibbsNode("children", ["bad", "good"], {(0,) : [0.6, 0.4], (1,) : [0.3, 0.7]}, [neighborhood])
    size = GibbsNode("size", ["small", "medium", "large"], {(): [0.33, 0.34, 0.33]}, [])
    schools = GibbsNode("schools", ["bad", "good"], {(0,) : {0.7, 0.3}, (1,) : {0.8, 0.2}}, [children])
    age = GibbsNode("age", ["old", "new"], {(0,): [0.3, 0.7], (1,) : {0.6, 0.4}, (2,): {0.9, 0.1}}, [location])
    price = GibbsNode("price", ["cheap", "ok", "expensive"], {
        (0, 0, 0, 0) : [0.5, 0.4, 0.1],
        (0, 0, 0, 1) : [0.4, 0.45, 0.15],
        (0, 0, 0, 2) : [0.35, 0.45, 0.2],
        (0, 0, 1, 0) : [0.4, 0.3, 0.3],
        (0, 0, 1, 1) : [0.35, 0.3, 0.35],
        (0, 0, 1, 2) : [0.3, 0.25, 0.45],
        (0, 1, 0, 0) : [0.45, 0.4, 0.15],
        (0, 1, 0, 1) : [0.4, 0.45, 0.15],
        (0, 1, 0, 2) : [0.35, 0.45, 0.2],
        (0, 1, 1, 0) : [0.25, 0.3, 0.45],
        (0, 1, 1, 1) : [0.2, 0.25, 0.55],
        (0, 1, 1, 2) : [0.1, 0.2, 0.7],
        (1, 0, 0, 0) : [0.7, 0.299, 0.001],
        (1, 0, 0, 1) : [0.65, 0.33, 0.02],
        (1, 0, 0, 2) : [0.65, 0.32, 0.03],
        (1, 0, 1, 0) : [0.55, 0.35, 0.1],
        (1, 0, 1, 1) : [0.5, 0.35, 0.15],
        (1, 0, 1, 2) : [0.45, 0.4, 0.15],
        (1, 1, 0, 0) : [0.6, 0.35, 0.05],
        (1, 1, 0, 1) : [0.55, 0.35, 0.1],
        (1, 1, 0, 2) : [0.5, 0.4, 0.1],
        (1, 1, 1, 0) : [0.4, 0.4, 0.2],
        (1, 1, 1, 1) : [0.3, 0.4, 0.3],
        (1, 1, 1, 2) : [0.3, 0.3, 0.4],
        (2, 0, 0, 0) : [0.8, 0.1999, 0.0001],
        (2, 0, 0, 1) : [0.75, 0.24, 0.01],
        (2, 0, 0, 2) : [0.75, 0.23, 0.02],
        (2, 0, 1, 0) : [0.65, 0.3, 0.05],
        (2, 0, 1, 1) : [0.6, 0.33, 0.07],
        (2, 0, 1, 2) : [0.55, 0.37, 0.08],
        (2, 1, 0, 0) : [0.7, 0.27, 0.03],
        (2, 1, 0, 1) : [0.64, 0.3, 0.06],
        (2, 1, 0, 2) : [0.61, 0.32, 0.07],
        (2, 1, 1, 0) : [0.48, 0.42, 0.1],
        (2, 1, 1, 1) : [0.41, 0.39, 0.2],
        (2, 1, 1, 2) : [0.37, 0.33, 0.3]
    }, [location, age, schools, size])
    node_list = [amenities, location, age, schools, size, children, price, neighborhood]

if __name__ == "__main__":
    main()