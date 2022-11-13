from module import Attacker

# load data
dtype = "imdb"
attack_type = "insertion"
attack_percentage = 0.05
path2txt = f"./results/ours/{dtype}.txt"
path2result = f"./{path2txt.split('.')[1]}-corrupted={attack_percentage}.txt"
constraint_kwargs = {'use': 0.95}

attacker = Attacker(attack_type, attack_percentage, path2txt, path2result, constraint_kwargs)
attacker.attack_sentence()
