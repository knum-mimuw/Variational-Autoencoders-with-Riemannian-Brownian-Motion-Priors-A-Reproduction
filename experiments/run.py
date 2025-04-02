from experiments.experiment import Experiment
from experiments.parser import get_parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    experiment = Experiment(args)

    if args.ckpt_path is None:
        experiment.train()
        experiment.eval()
    else:
        experiment.eval(args.ckpt_path)
