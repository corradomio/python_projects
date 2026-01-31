from datetime import datetime
from datetimex import relativedifference, relativeperiods, clip_date


def main():
    freq = 'W'
    this_datetime = clip_date(datetime(2024, 7, 13), freq='D')
    that_datetime = clip_date(datetime.now(), freq='D')

    print("  start", this_datetime)
    print("    end", that_datetime)

    periods = relativedifference(that_datetime, this_datetime, freq=freq)
    print("periods", periods)
    print("   date", this_datetime + relativeperiods(periods=periods, freq=freq))



if __name__ == "__main__":
    main()
