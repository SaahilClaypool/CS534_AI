import matplotlib.pyplot as plt

import gibbs
from typing import Type, Sequence, Dict

def batch_simulate(node_list: Sequence['gibbs.GibbsNode'], iterationNumbs: Sequence[int], dropnum: int):
    """
    return each simulation statistics, with both with drops and without drops
    """
    x = node_list[0]

    # dict [query][prob]
    drop_probs = dict()
    nodrop_probs = dict()

    for q in node_list: 
        drop_probs[q.get_name()] = {}
        nodrop_probs[q.get_name()] = {}

    for it in iterationNumbs: 
        # do without drops
        drops = gibbs.simulate(node_list, it, 0)
        for q in drops: 
            drop_probs[q.get_name()][it] = q.get_independent_probability()

        # do with drops
        nodrops = gibbs.simulate(node_list, it, 0)
        for q in nodrops: 
            nodrop_probs[q.get_name()] = q.get_independent_probability()
    
    return {"drop": drop_probs, "nodrop": nodrop_probs}

def make_plots(probs):
    """
    plot each probability for each query, saving to a file in the /img folder

    Args:
        probs: dict of {'drops': {'query', {'state', 1.0 (prob),...} } 'nodrop'...}
    """
    drops = probs["drop"]
    nodrops = probs["nodrop"]
    for query, its in drops.items(): 
        x = []
        y = {}
        for it, result in its.items():
            x.append(it)
            for state, prob in result.items():
                if(state not in y.keys()):
                    y[state] = []
                y[state].append(prob)

        for k,v in y.items():
            print(k,len(v))
        # for each query, make a plot
        for state, probs in y.items():
            plt.plot(x, probs)
        plt.show()
        
    pass

def main():
    print("hello world")
    node_list = gibbs.setup()
    probs = batch_simulate(node_list, range(40, 4020, 100), 50)
    make_plots(probs)
    ## Plots: for each thing, 



if __name__ == "__main__":
    main()
