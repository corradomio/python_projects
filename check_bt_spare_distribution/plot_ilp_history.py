import matplotlib.pyplot as plt
from matplotlib import rcParams
from path import Path as path


GUROBI_HOME=path("3600s")


# ---------------------------------------------------------------------------

class GurobiLog:

    HEADER_END = "Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time"

    def __init__(self, log_file=None):
        self.log_file = log_file
        self.lines: list[str] = []
        self.line: int = 0

        self.primal = []
        self.dual = []
    # end

    def parse(self, log_file=None):
        if log_file is not None:
            self.log_file = log_file

        self._load_file()
        self._skip_header()
        self._parse_history()

        pass
    # end

    def _load_file(self):
        with open(self.log_file) as f:
            self.lines = f.read().splitlines()
    # end

    def _skip_header(self):
        #
        #    Nodes    |    Current Node    |     Objective Bounds      |     Work
        # Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time
        #
        #      0     0 5370.85351    0   47          - 5370.85351      -     -    0s
        # H    0     0                    8811.2241467 5370.85351  39.0%     -    0s
        #.

        n_lines = len(self.lines)
        for i in range(n_lines):
            line = self.lines[i].strip()
            if line != self.HEADER_END:
                continue

            # set to the FIRST line of the history, skiupping the empty line
            self.line = i+2
            break
        pass
    # end

    def _parse_history(self):
        n_lines = len(self.lines)

        # skip the FIRST list of history because in some coases it is not complete
        for i in range(self.line+1, n_lines):
            line = self.lines[i].strip()
            if len(line) == 0:
                break

            # H    0     0                    8147.7749743 5200.22099  36.2%     -    0s
            # 204526 84869 5984.95185   41  146 6073.84018 5901.71597  2.83%  40.0   55s

            parts = line.split()
            Incumbent = float(parts[-5])
            BestBound = float(parts[-4])

            self.primal.append(Incumbent)
            self.dual.append(BestBound)
            pass
        # end
    # end
# end


def nw_of(s: str) -> int:
    b = s.find('-')
    return int(s[b+1:])


def plot_log(log_file: path):
    # print(log_file)

    nw = nw_of(log_file.stem)
    time = log_file.parent.stem

    gl = GurobiLog(log_file)
    gl.parse()

    pwidth = 6
    pheight = 3  # 3.5

    plt.clf()
    plt.gcf().set_size_inches(pwidth, pheight)

    plt.plot(gl.primal)
    plt.plot(gl.dual)
    plt.xlabel("steps")
    plt.ylabel("solution value")
    plt.legend(["primal", "dual"])
    if nw == 50:
        plt.title(f"(a)   ILP history (N={nw})", fontdict={
            'fontsize': 14
        })
    elif nw == 100:
        plt.title(f"(b)   ILP history (N={nw})", fontdict={
            'fontsize': 14
        })
    else:
        plt.title(f"ILP history (N={nw})")
    plt.tight_layout(pad=0.5)

    fname = f"results_plots_ilp/ilp_history-{time}-{nw:03}.png"
    plt.savefig(fname, dpi=300)
    print(fname)
    pass
# end


def plot_all_ilp_plots(all_gl: dict[int, GurobiLog]):
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(6, 4))

    c = -1
    for nw in [50,60,70,80,90,100]:
        c += 1
        gl = all_gl[nw]

        ax = axs[c//3, c%3]
        plt.sca(ax)
        plt.plot(gl.primal)
        plt.plot(gl.dual)
        if c == 0:
            plt.legend(["primal", "dual"])

        plt.text(0.75, 0.05, f"N={nw}",fontsize=8, transform=ax.transAxes)
    # end

    plt.tight_layout(pad=1.75)
    # plt.tight_layout()

    for i in range(2):
        axs[i, 0].set_ylabel("solution_value")

    for j in range(3):
        axs[1,j].set_xlabel("steps")

    fname = f"results_plots_ilp/ilp_history-7200s-all.png"
    plt.savefig(fname, dpi=300)
    pass
# end

# ---------------------------------------------------------------------------

def main():
    # GUROBI_HOME=path("3600s")
    # for log in GUROBI_HOME.files("*.log"):
    #     plot_log(log)
    # # end

    all_logs = {}

    GUROBI_HOME=path("7200s")
    for log_file in GUROBI_HOME.files("*.log"):
        plot_log(log_file)
        nw = nw_of(log_file.stem)
        gl = GurobiLog(log_file)
        gl.parse()

        all_logs[nw] = gl

    plot_all_ilp_plots(all_logs)
    # end
# end


if __name__ == "__main__":
    main()
