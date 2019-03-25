import pickle
from algorithms import TournamentSelection

op1 = TournamentSelection()
op2 = TournamentSelection(n_parents=4, t_size=5, maximize=True)

print(op1.export())
print(op2.export())

with open('op.pickle', 'wb') as f:
    pickle.dump([op1, op2], f)
    print('saved')

op1 = None
op2 = None

with open('op.pickle', 'rb') as f:
    data = pickle.load(f)
    op1 = data[0]
    op2 = data[1]
    print('loaded')

print(op1.export())
print(op2.export())
