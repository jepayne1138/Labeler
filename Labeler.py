import sys
import labeler.views as views
from labeler.models.config import Config


if __name__ == '__main__':
    Config.initialize_config()
    sys.exit(views.exec_app(sys.argv))
