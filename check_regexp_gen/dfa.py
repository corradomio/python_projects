from typing import Optional, cast, Union

STATE_TYPE = int
ELEMENT_TYPE = str


class Automata:
    def __init__(self, **aprops):
        """

        :param aprops: automata properties
        """
        self.props = aprops
        self._initial_state: STATE_TYPE = 0
        self._states: set[STATE_TYPE] = {0}
        self._alphabet: set[ELEMENT_TYPE] = set()
        self._final_states: set[STATE_TYPE] = set()

        # {
        #    (s, e): s'
        # }

        self._transitions: dict[tuple[STATE_TYPE, ELEMENT_TYPE], STATE_TYPE] = {}
    # end

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state: STATE_TYPE):
        self._initial_state = state

    @property
    def final_states(self):
        return self._final_states

    def add_final_state(self, state: STATE_TYPE):
        self._final_states.add(state)
        return self

    def add_transitions(self, sequence: Union[str, list[ELEMENT_TYPE]], *, state: Optional[STATE_TYPE] = None,
                        final_state=True):
        """
        Add a list of transitions starting from the initial state (or the specified state) and using the elements
        in sequence
        :param sequence: sequence of elements
        :param state: initial state, if specified
        :param final_state: if the last state is a final state
        :return:
        """
        if state is None:
            state = self.initial_state
        for elt in sequence:
            state = self.add_transition(state, elt, None)
        if final_state:
            self.add_final_state(state)
        return self

    def add_transition(self, state: STATE_TYPE, elt: ELEMENT_TYPE, next_state: Optional[STATE_TYPE]):
        """
        Add a single transition.
        If the next state is not specified, it is created a new one
        If the transition (state, elt) already exists, it is returned the registered next state

        :param state: current state
        :param elt:current element
        :param next_state: next state. If it is None, it is generated a new one
        :return: the next state
        """
        next_state = self._find_transition(state, elt)
        if next_state:
            return next_state

        if elt not in self._alphabet:
            self._alphabet.add(elt)

        if next_state is None:
            next_state = len(self._states)
            self._states.add(next_state)

        self._add_transition(state, elt, next_state)
        return next_state

    def _find_transition(self, state, elt):
        return self._transitions.get((state, elt))

    def _add_transition(self, state, elt, next_state):
        self._transitions[(state, elt)] = next_state

    def _all_transitions(self):
        return self._transitions.keys()

    def transition(self, state: STATE_TYPE, elements: ELEMENT_TYPE) -> STATE_TYPE:
        cstate = state
        for elt in elements:
            cstate = self._find_transition(cstate, elt)
        return cstate

    def next_states(self, state: STATE_TYPE) -> tuple[set[STATE_TYPE], set[ELEMENT_TYPE]]:
        nstates = set()
        salphab = set()
        for t in self._all_transitions():
            if state == t[0]:
                salphab.add(t[1])
                nstates.add(self._find_transition(t[0], t[1]))
        return nstates, salphab

    def match(self, sequence: Union[str, list[ELEMENT_TYPE]]) -> bool:
        """
        Check if the automata matches the sequence
        :param sequence:
        :return:
        """
        state = self.initial_state
        for elt in sequence:
            state = self._find_transition(state, elt)
            if state is None:
                return False
        if state in self._final_states:
            return True
        else:
            return False
    # end

    def __repr__(self):
        Q = self._states
        A = self._alphabet
        q0 = self.initial_state
        F = self._final_states
        T = self._all_transitions()
        return f"(|Q|={len(Q)}, |A|={len(A)}, qo={q0}, |F|={len(F)}, |T|={len(T)})"
    # end
# end
