from netx.dfa import Automata
import stdlib.csvx as csvx


def main():
    invoices = csvx.load_csv("generatedInvoices_v4.csv", skiprows=1, dtype=[None] * 3 + [str, str])

    invoice_dfas: dict[str, Automata] = {}

    print("create DFAs ...")
    for invoice in invoices:
        i_id = invoice[0]
        name = invoice[1]

        if name not in invoice_dfas:
            print(f"... {name}")
            dfa = Automata(name=name)
            invoice_dfas[name] = dfa
        else:
            dfa = invoice_dfas[name]

        print(f"... .... {i_id}")
        dfa.add_transitions(i_id)
    # end

    print("list DFAs ...")
    for name in invoice_dfas:
        print(f"... {name}:  {invoice_dfas[name]}")

    print("validate DFAs ...")
    for invoice in invoices:
        i_id = invoice[0]
        name = invoice[1]

        dfa = invoice_dfas[name]

        if not dfa.match(i_id):
            print(f"ERROR: DFA {name} doesn't match {i_id}")

    print("next_states DFAs ...")
    for name in invoice_dfas:

        dfa = invoice_dfas[name]
        initial_state = dfa.initial_state
        next_states, elements = dfa.next_states(initial_state)

        print(f"... {name}: {initial_state} -> ({next_states}, {elements})")

    print("done")
    pass


if __name__ == "__main__":
    # logging.config.fileConfig('logging_config.ini')
    # log = logging.getLogger("root")
    # log.info("Logging system configured")
    main()

