import subprocess
import argparse
import itertools

Models = ['gcn','sage','gat','gatv2','hyper-gcn','hyper-gat'] # 'hybrid-gcn','hybrid-sage' have to be used with graph sampling
Datasets = ["grand_ArteryAorta","grand_Breast","grand_Vagina",
            "grand_ArteryCoronary","grand_ColonAdenocarcinoma",
            "grand_Sarcoma","grand_Liver","grand_TibialNerve",
            "grand_KidneyCarcinoma","grand_Spleen",
            "musae_Twitch_ES","musae_Twitch_FR","musae_Twitch_DE",
            "musae_Twitch_EN","musae_Twitch_PT","musae_Twitch_RU",
            "musae_Facebook","musae_Github","musae_Wiki_chameleon",
            "musae_Wiki_crocodile","musae_Wiki_squirrel"]
# the total number of experiment: model num * datasets num * task type = 6 (+4) * 19 * 2

def run_train(*args):
    content = ['python', 'hg.py', 'train']
    content.extend(args[0])
    subprocess.call(content)

def run_test(*args):
    content = ['python', 'hg.py', 'eval']
    content.extend(args[0])
    subprocess.call(content)

def main():
    parser = argparse.ArgumentParser(description="A script that takes a single argument train or test")
    parser.add_argument('-t', '--type', type=str, required=True, help='train or test')
    parser.add_argument('-m', '--max_epoch', type=int, required=False, default=50, help='max epoch')
    args = parser.parse_args()
    combinations = list(itertools.product(Models, Datasets))
    if args.type == 'train':
        for model, dataset in combinations:
            run_train([dataset,model,f'-m={args.max_epoch}',
                       f'-save=checkpoints/{dataset}_{model}'])
    elif args.type == 'test':
        for model, dataset in combinations:
            run_test([dataset,model,f'-load=checkpoints/{dataset}_{model}/'])

if __name__ == '__main__':
    main()
