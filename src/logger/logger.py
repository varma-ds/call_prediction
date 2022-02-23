import logging
import logging.config
from pathlib import Path
import json

def setup_logger(save_dir, log_config='src/logger/logger_config.json'):
    log_config = Path(log_config)
    with open(log_config) as lcf:
        config = json.load(lcf)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename']= str(save_dir/handler['filename'])

        logging.config.dictConfig(config)

def get_logger(name):
    return logging.getLogger(name)


if __name__ == '__main__':
    setup_logger(Path('.'),)
    logger = logging.getLogger(__name__)
    logger.info('My name is vijay')




