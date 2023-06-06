# Modules for Data Preprocessing

## Sampler

```python
from hg.datasets import Facebook, HypergraphSAINTNodeSampler
# download data to the path 'data/facebook'
data = Facebook('data/facebook')
print(data[0]) 
# e.g. Data(x=[22470, 128], edge_index=[2, 342004], y=[22470], hyperedge_index=[2, 2344151], num_hyperedges=236663)

# create a sampler which sample 1000 nodes from the graph for 5 times
sampler = HypergraphSAINTNodeSampler(data[0],batch_size=1000,num_steps=5)
batch = next(iter(sampler))
print(batch)  
# e.g. Data(num_nodes=918, edge_index=[2, 7964], hyperedge_index=[2, 957528], num_hyperedges=210718, x=[918, 128], y=[918])
```


# Modules for Training and Evaluation
## Train


## Evaluate