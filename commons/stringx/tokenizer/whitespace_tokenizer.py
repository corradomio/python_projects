from . import utils
from .delimiter_tokenizer import DelimiterTokenizer


class WhitespaceTokenizer(DelimiterTokenizer):
    """Segments the input string using whitespaces then returns the segments as tokens. 
    
    Currently using the split function in Python, so whitespace character refers to 
    the actual whitespace character as well as the tab and newline characters. 

    Args:
        return_set (boolean): A flag to indicate whether to return a set of
                              tokens instead of a bag of tokens (defaults to False).
                              
    Attributes:
        return_set (boolean): An attribute to store the flag return_set. 
    """
    
    def __init__(self, return_set=False):
        super(WhitespaceTokenizer, self).__init__([' ', '\t', '\n'], return_set)

    def tokenize(self, input_string):
        """Tokenizes input string based on white space.

        Args:
            input_string (str): The string to be tokenized. 

        Returns:
            A Python list, which is a set or a bag of tokens, depending on whether return_set is True or False. 

        Raises:
            TypeError : If the input is not a string.

        Examples:
            >>> ws_tok = WhitespaceTokenizer() 
            >>> ws_tok.tokenize('data science')
            ['data', 'science']
            >>> ws_tok.tokenize('data        science')
            ['data', 'science']
            >>> ws_tok.tokenize('data\tscience')
            ['data', 'science']
            >>> ws_tok = WhitespaceTokenizer(return_set=True) 
            >>> ws_tok.tokenize('data   science data integration')
            ['data', 'science', 'integration']
        """
        utils.tok_check_for_none(input_string)
        utils.tok_check_for_string_input(input_string)

        token_list =  list(filter(None, input_string.split()))

        if self.return_set:
            return utils.convert_bag_to_set(token_list)

        return token_list

    def set_delim_set(self, delim_set):
        raise AttributeError('Delimiters cannot be set for WhitespaceTokenizer')
