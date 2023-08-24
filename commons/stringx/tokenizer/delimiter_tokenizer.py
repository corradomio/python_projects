import re

from . import utils
from .tokenizer import Tokenizer


class DelimiterTokenizer(Tokenizer):
    """Uses delimiters to find tokens, as apposed to using definitions. 
    
    Examples of delimiters include white space and punctuations. Examples of definitions include alphabetical and qgram tokens. 

    Args:
        delim_set (set): A set of delimiter strings (defaults to space delimiter).
        return_set (boolean): A flag to indicate whether to return a set of
                              tokens instead of a bag of tokens (defaults to False).
                              
    Attributes: 
        return_set (boolean): An attribute to store the value of the flag return_set.
    """

    def __init__(self, 
                 delim_set=set([' ']), return_set=False):
        self.__delim_set = None
        self.__use_split = None
        self.__delim_str = None
        self.__delim_regex = None
        self._update_delim_set(delim_set)
        super(DelimiterTokenizer, self).__init__(return_set)

    def tokenize(self, input_string):
        """Tokenizes input string based on the set of delimiters.

        Args:
            input_string (str): The string to be tokenized. 

        Returns:
            A Python list which is a set or a bag of tokens, depending on whether return_set flag is set to True or False. 

        Raises:
            TypeError : If the input is not a string.

        Examples:
            >>> delim_tok = DelimiterTokenizer() 
            >>> delim_tok.tokenize('data science')
            ['data', 'science']
            >>> delim_tok = DelimiterTokenizer(['$#$']) 
            >>> delim_tok.tokenize('data$#$science')
            ['data', 'science']
            >>> delim_tok = DelimiterTokenizer([',', '.']) 
            >>> delim_tok.tokenize('data,science.data,integration.')
            ['data', 'science', 'data', 'integration']
            >>> delim_tok = DelimiterTokenizer([',', '.'], return_set=True) 
            >>> delim_tok.tokenize('data,science.data,integration.')
            ['data', 'science', 'integration']

        """
        utils.tok_check_for_none(input_string)
        utils.tok_check_for_string_input(input_string)
    
        if self.__use_split:
            token_list = list(filter(None,
                                     input_string.split(self.__delim_str)))
        else:
            token_list = list(filter(None,
                                     self.__delim_regex.split(input_string)))

        if self.return_set:
            return utils.convert_bag_to_set(token_list)

        return token_list

    def get_delim_set(self):
        """Gets the current set of delimiters.
        
        Returns:
            A Python set which is the current set of delimiters. 
        """
        return self.__delim_set

    def set_delim_set(self, delim_set):
        """Sets the current set of delimiters.
        
        Args:
            delim_set (set): A set of delimiter strings.
        """
        return self._update_delim_set(delim_set)

    def _update_delim_set(self, delim_set):
        if not isinstance(delim_set, set):
            delim_set = set(delim_set)
        self.__delim_set = delim_set
        # if there is only one delimiter string, use split instead of regex
        self.__use_split = False
        if len(self.__delim_set) == 1:
            self.__delim_str = list(self.__delim_set)[0]
            self.__use_split = True
        else:
            self.__delim_regex = re.compile('|'.join(
                                     map(re.escape, self.__delim_set)))
        return True
