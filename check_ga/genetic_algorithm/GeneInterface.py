from abc import ABC, abstractmethod


class GeneInterface(ABC):
    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def mutate(self):
        pass

