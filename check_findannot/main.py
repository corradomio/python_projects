class FindBracket:

    def __init__(self, s, p=0):
        self.s = s
        self.p = p
        self.n = len(s)

        # in string
        self.c = 0
        # nested [...[...]...]
        self.d = 0

    def find(self):
        i = 0
        while i < self.n:
            ch = self.ch()
            if self.c:
                if ch == '"' and self.nc() == '"':
                    self.next()
                    self.next()
                elif ch == '"':
                    self.c = 0
                    self.next()
                elif ch == '\\':
                    self.next()
                    self.next()
                else:
                    self.next()
            else:
                if ch == '[':
                    self.d += 1
                    self.next()
                elif ch == ']':
                    self.d -= 1
                    if self.d == 0:
                        break
                    else:
                        self.next()
                elif ch == '"':
                    self.c = 1
                    self.next()
                else:
                    self.next()
                # end
            # end
            i += 1
        # end
        return self.p
    # end

    def ch(self):
        return self.s[self.p] if self.p < self.n else 0

    def nc(self):
        return self.s[self.p+1] if self.p < (self.n-1) else 0

    def next(self):
        self.p = self.p+1 if self.p < self.n else self.n
# end


def find_bracket(s, p=0):
    fb = FindBracket(s, p)
    return fb.find()
# end


def main():
    s = """
    [Cmdlet(VerbsCommon.Remove, "Job", SupportsShouldProcess = true, DefaultParameterSetName = JobCmdletBase.SessionIdParameterSet,
        HelpUri = "https://go.microsoft.com/fwlink/?LinkID=2096868")]
    [OutputType(typeof(Job), ParameterSetName = new string[] { JobCmdletBase.JobParameterSet })]
    public class RemoveJobCommand : JobCmdletBase, IDisposable
    """
    b = s.find('[', 0)
    e = find_bracket(s, b)
    print(f"'{s[b+1:e]}'")

    b = s.find('[', e+1)
    e = find_bracket(s, b)
    print(f"'{s[b+1:e]}'")


if __name__ == '__main__':
    main()
