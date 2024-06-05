from beacon.scope import Scope
import argparse

scope = Scope(use_external_parser=True)
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', default=16, type=int, help='batch size.')


@scope.observe(default=True)
def default(config):
    config.lr = 0.1


@scope.manual
def default_manual(config):
    config.lr = 'learning rate.'


@scope
def main(config):
    pass


if __name__ == '__main__':
    parser.parse_args()
    main()
