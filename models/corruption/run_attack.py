from module import Attacker
from config import CorruptionArgs

parser = CorruptionArgs()
args = parser.parse_args()

# load data
attack_type = args.attack_type
attack_percentage = args.attack_pct
method = args.target_method
path2embed = args.path2embed

if args.exp_name:
    path2result = f"./{path2embed}/{args.exp_name}-corrupted={attack_percentage}.txt"
else:
    path2result = f"./{path2embed.split('.')[0]}-corrupted={attack_percentage}.txt"
constraint_kwargs = {'use': 0.95, 'num_sentence': args.num_sentence}
attacker = Attacker(attack_type, attack_percentage, path2embed, path2result, constraint_kwargs)
attacker.attack_sentence()
