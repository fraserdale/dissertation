import pickle
with open ('reinforcement_eval.pkl', 'rb') as fp:
    itemlist = pickle.load(fp)
    
print(itemlist)