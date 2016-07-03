import sys
import labeler.views as views
import labeler.modules.cmd_parser as cmd_parser
from labeler.modules.config import Config
from labeler.modules.word_bag import WordBag


if __name__ == '__main__':
    Config.initialize_config()
    args = cmd_parser.parse_arguments(sys.argv[1:])
    # bag = WordBag()
    # import json
    # with open('labeled_files/labeled/e34256a.json', 'r') as json_file:
    #     data = json.load(json_file)
    # bag.label_frequency(data)
    sys.exit(views.exec_app(sys.argv))
