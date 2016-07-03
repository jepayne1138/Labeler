import argparse


DESCRIPTION = 'Cleans and standardizes lists of mailing addresses.'


def parse_arguments(args):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    return parser.parse_args(args=args)
