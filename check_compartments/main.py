# This is a sample Python script.

from datetime import date
from typing import List, Optional, Dict


class UserStatus:

    def __init__(self):
        self._s = 1.        # susceptible
        self._r = 0.        # recovered
        self._e = 0.        # exposed
        self._i = 0.        # infectious
        self._d = date()
        self._d_exposed: Optional[date] = None    # latest exposed day

        self._days_l = 5    # latent days
        self._days_r = 15   # recovered days
    # end

    def encounter(self, iprob: float, d: date):
        if self._d_exposed is None:
            self._d_exposed = d
        elif (d - self._d_exposed).days > self._days_l:
            self._d_exposed = d

        exposed = iprob*self._s
        self._s -= exposed
        self._e += exposed
    # end

    def check_day(self, d: date):
        if self._d_exposed is None:
            return
        days = (d - self._d_exposed).days
        if days > self._days_r:
            self._r += self._i
            self._i = 0.
            self._d_exposed = None
        elif days > self._days_l:
            self._i += self._e
            self._e = 0.
    # end

    def infectious(self) -> float:
        return self._i
    # end

    @property
    def status(self) -> Dict[str, float]:
        return {"s": self._s, "r": self._r, "e": self._e, "i": self._i}

# end


def main():
    us = UserStatus()

    print(us.status)

    us.encounter(.25, date(2021, 1, 1))
    us.encounter(.25, date(2021, 1, 4))
    for day in range(1, 28):
        us.check_day(date(2021, 1, day))
        print(us.status)
    us.encounter(.25, date(2021, 2, 1))
    for day in range(1, 28):
        us.check_day(date(2021, 2, day))
        print(us.status)
# end


if __name__ == "__main__":
    main()

