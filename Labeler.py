import sys
import labeler.views as views
from labeler.models.config import Config
from labeler.models.word_bag import WordBag


if __name__ == '__main__':
    Config.initialize_config()
    bag = WordBag()
    # import json
    # with open('labeled_files/labeled/e34256a.json', 'r') as json_file:
    #     data = json.load(json_file)
    # bag.label_frequency(data)
    # sys.exit(views.exec_app(sys.argv))
