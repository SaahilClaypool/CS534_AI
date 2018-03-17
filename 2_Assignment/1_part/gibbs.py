from typing import Type, Sequence, Dict
import sys
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
        self.value = random.randint(0, len(value_names)-1)#initialize to random value
        self.max_value = len(self.value_names)-1
        self.is_fixed = False
        self.children = [] #initialize children as empty, add afterwards
        self.value_counts = {}
        for v in value_names: 
            self.value_counts[v] = 0
        
    
    def __str__(self):
        s = "{}: ".format(self.get_name())
        for n in self.value_names: 
            s += "\n\t{}: {}".format(n, self.value_counts[n])
        return s 
                
    def __repr__(self):
        s = "{}: ".format(self.get_name())
        for n in self.value_names: 
            s += "\n\t{}: {}".format(n, self.value_counts[n])
        return s 
    
    def get_independent_probability(self) -> Dict[str, float]:
        """
        get the final probability of each state given the counts
        """
        total = sum(self.value_counts.values())
        if(total == 0):
            return {}
        probs = {}
        for v,k in self.value_counts.items(): 
            probs[v] = k / total
        return probs



    def set_value(self, value: int, iterationNumber: int, throwout) :
        """
        Set the given value, track that the value was changed if it is larger than the iteration number
        (this will effectively allow droppping)
        """
        self.value = value
        if (iterationNumber > throwout-1):
            self.value_counts[self.value_names[value]] += 1
        
    def get_name(self) -> str:
        return self.name

    def set_fixed(self, f : bool, val_str : str):
        self.is_fixed = f
        self.value = self.value_names.index(val_str)

    def set_children(self, children : Sequence['GibbsNode']):
        self.children = children

    def get_string_value(self) -> str:
        return self.value_names[self.value]

    def get_current_value(self) -> int:
        return self.value

    def set_value_random(self):
        self.value = random.randint(0, self.max_value)

    def get_cond_prob_given_parent(self, p_node : 'GibbsNode', p_val : int) -> float:
        """
        Determine the probability of this node given a certain configuration of a parent node
        i.e. P (this_value | parent_value)
        """
        p_vals = []
        for p in self.parents:
            if p == p_node:
                p_vals.append(p_val)
            else:
                p_vals.append(p.get_current_value())
        parent_values = tuple(p_vals)
        probabilities = self.p_dict[parent_values]
        return probabilities[self.value]

    def update_node(self, iterationNumber: int, throwout: int):
        """
        Update the current node based on its Markov blanket and basic probability distribution
        """
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
        probabilities =  list(self.p_dict[parent_values])

        #get conditional probabilities from children
        if self.children:
            for c in self.children:
                for i in range(0, self.max_value+1):
                    # multiply the probability of the i'th state (given parents) by the probability
                    # of the i'th state given the child state. Do for each child
                    probabilities[i] = probabilities[i] * c.get_cond_prob_given_parent(self, i)

        #normalize
        normalizer = 1.0 / sum(probabilities)
        for i in range(0, self.max_value+1):
            probabilities[i] = probabilities[i] * normalizer

        randv = random.random()
        ptotal = 0
        #iterate through and add each probability, use this running total to determine the new value of this node
        for i in range(0, self.max_value+1):
            ptotal += probabilities[i]
            if randv < ptotal:
                # self.value = i
                self.set_value(i, iterationNumber, throwout)
                return

def setup():
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
    schools = GibbsNode("schools", ["bad", "good"], {(0,) : [0.7, 0.3], (1,) : [0.8, 0.2]}, [children])
    age = GibbsNode("age", ["old", "new"], {(0,): [0.3, 0.7], (1,) : [0.6, 0.4], (2,): [0.9, 0.1]}, [location])
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
    amenities.set_children([location])
    neighborhood.set_children([location, children])
    children.set_children([schools])
    location.set_children([age, price])
    size.set_children([price])
    schools.set_children([price])
    age.set_children([price])
    node_list = [amenities, location, age, schools, size, children, price, neighborhood]
    return node_list

def main():

    node_list = setup()
    #set up evidence and node of interest based on commandline arguments
    #no robust error checking of commandline arguments takes place
    target_node_name = sys.argv[1]
    #default values
    update_count = 10000
    drop_count = 1000
    i = 2
    while i < len(sys.argv):
        #set update count
        if sys.argv[i] == "-u":
            i += 1
            update_count = int(sys.argv[i])
        #set drop count
        elif sys.argv[i] == "-d":
            i += 1
            drop_count = int(sys.argv[i])
        #otherwise set evidence node
        else:
            eq_i = sys.argv[i].find("=")
            evidence_name = sys.argv[i][:eq_i]
            evidence_val = sys.argv[i][eq_i+1:]
            for n in node_list:
                if n.get_name() == evidence_name:
                    n.set_fixed(True, evidence_val)
        i += 1

    simulated_nodes = simulate(node_list, update_count, drop_count)
    for n in simulated_nodes:
        if n.get_name() == target_node_name:
            print("Probabilities of ", target_node_name, " : ", n.get_independent_probability())
    
def simulate(node_list: Sequence['GibbsNode'], iterations: int, throwout: int) -> Sequence['GibbsNode']:
    """
    Choose
    Update
    Iterate
    """
    mutables : Sequence['GibbsNode'] = list(filter(lambda n : not n.is_fixed, node_list))
    for i in range(iterations): 
        selection = random.choice(mutables)
        selection.update_node(i, throwout)
    return mutables
        

if __name__ == "__main__":
    main()
