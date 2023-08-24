import re

from . import utils
from .definition_tokenizer import DefinitionTokenizer


class AlphabeticTokenizer(DefinitionTokenizer):
    """Returns tokens that are maximal sequences of consecutive alphabetical characters.
    
    Args:
        return_set (boolean): A flag to indicate whether to return a set of tokens instead of a bag of tokens (defaults to False).
        
    Attributes: 
        return_set (boolean): An attribute that stores the value for the flag return_set. 
    """
    
    def __init__(self, return_set=False):
        self.__al_regex = re.compile('[a-zA-Z]+')
        super(AlphabeticTokenizer, self).__init__(return_set)

    def tokenize(self, input_string):
        """Tokenizes input string into alphabetical tokens.
        
        Args:
            input_string (str): The string to be tokenized.

        Returns:
            A Python list, which represents a set of tokens if the flag return_set is True, and a bag of tokens otherwise. 

        Raises:
            TypeError : If the input is not a string.

        Examples:
            >>> al_tok = AlphabeticTokenizer()
            >>> al_tok.tokenize('data99science, data#integration.')
            ['data', 'science', 'data', 'integration']
            >>> al_tok.tokenize('99')
            []
            >>> al_tok = AlphabeticTokenizer(return_set=True) 
            >>> al_tok.tokenize('data99science, data#integration.')
            ['data', 'science', 'integration']
        """
        utils.tok_check_for_none(input_string)
        utils.tok_check_for_string_input(input_string)

        token_list = list(filter(None, self.__al_regex.findall(input_string)))

        if self.return_set:
            return utils.convert_bag_to_set(token_list)

        return token_list
