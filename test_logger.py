import logging
# # import mylib
# import utils
from utils import *

# logger = logging.getLogger('__name__')
# logger.setLevel('DEBUG')

def get_logger_handler(path):
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)

    f_handler = logging.FileHandler(path)
    f_handler.setLevel(logging.DEBUG)

    return c_handler, f_handler

# def main():
    
#     # logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
#     # logger.getLogger().addHandler(logging.StreamHandler())
#     logger = logging.getLogger('spam_application')
#     c_handler, f_handler = get_logger_handler('myapp.log')
#     logger.addHandler(c_handler)
#     logger.addHandler(f_handler)

#     logger.debug('Started')
#     test_logging()
#     logger.debug('Finished')
    # logging.warning('Watch out!')  # will print a message to the console

# if __name__ == '__main__':

#     main()

# logger = logging.getLogger('__name__')
# logger.setLevel(logging.DEBUG)
# # logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
# # logger.getLogger().addHandler(logging.StreamHandler())
# # logger = logging.getLogger('spam_application')
# c_handler, f_handler = get_logger_handler('myapp.log')
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)

# logger.info('Started')
# test_logging()
# logger.info('Finished')




# import logging
# import auxiliary_module

# create logger with 'spam_application'
logger = logging.getLogger('__name__')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('myapp.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


logger.info('Started')
test_logging()
logger.info('Finished')
# logger.info('creating an instance of auxiliary_module.Auxiliary')
# logger.info('created an instance of auxiliary_module.Auxiliary')
# logger.info('calling auxiliary_module.Auxiliary.do_something')
# logger.info('finished auxiliary_module.Auxiliary.do_something')
# logger.info('calling auxiliary_module.some_function()')
# auxiliary_module.some_function()
# logger.info('done with auxiliary_module.some_function()')