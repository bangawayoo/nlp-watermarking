from module import Attacker

# load data
dtype = "imdb"
attack_type = "insertion"
attack_percentage = 0.10
path2txt = f"./results/context-ls-{dtype}-0-100.txt"
path2result = f"results/context-ls-{dtype}-corrupted={attack_percentage}.txt"
constraint_kwargs = {'use': 0.95}

attacker = Attacker(attack_type, attack_percentage, path2txt, path2result, constraint_kwargs)
attacker.attack_sentence()
