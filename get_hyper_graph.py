from datasets import Grand, Amazon, Facebook, GitHub, GraphStats
data = Grand('data/grand','Leukemia')
#print(GraphStats(data).get_all_stats())
mask_split(data)
