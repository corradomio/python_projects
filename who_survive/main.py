#
# https://betterprogramming.pub/100-people-stand-in-line-for-each-turn-a-person-in-an-odd-position-is-randomly-killed-e9145c997e1d
#


import random
import matplotlib.pyplot as plt


# simulate prograss
# One hundred people stand in line,
# for each turn a person in an odd position in randomly killed.
# Who has the greatest probability to survive to last?
# return which people survive to last
def simulate() -> int:
    # create a list of 100 people
    # etc. people[0] = 1 means the first people
    # people[99] = 100 means the last people
    people = list(range(1, 101))

    for i in range(99):
        # random choose a people in an odd position
        choose_number = random.randrange(1, len(people) + 1, 2)
        # kill the people in an odd position
        people.pop(choose_number - 1)

    # return the people survive to last
    return people[0]


# 100 element list, save the survive times of each people survive, default is 0
# etc. survive_times[0] = 10 means the first people survive 10 times
# survive_times[99] = 50 means the last people survive 50 times
survive_times = [0] * 100

SIMULATE_TIMES = 1000_000
# simulate some times
for i in range(SIMULATE_TIMES):
    if i % 1000 == 0:
        print(".", end="")
        if i % 100_000 == 0:
            print()
    survive_number = simulate()
    # add the survive times of each people
    survive_times[survive_number - 1] += 1
print()


if __name__ == '__main__':
    print(survive_times)

    # draw the survive times of each people

    plt.bar(range(1, 101), survive_times)
    plt.xlabel('People Number')
    plt.ylabel('Survive times')
    plt.title('Survive times')
    # save the figure
    plt.savefig('survive_times.png')
    # clear the figure
    plt.clf()

    # draw the survive rate of each people
    survive_rate = [i / SIMULATE_TIMES for i in survive_times]
    plt.bar(range(1, 101), survive_rate)
    plt.xlabel('People Number')
    plt.ylabel('Survive Rate')
    plt.title('Survive Rate')
    # save the figure
    plt.savefig('survive_rate.png')
    # clear the figure
    plt.clf()

    # draw the survive times of each people in odd position
    survive_times_odd = survive_times[::2]
    plt.bar(range(1, 51), survive_times_odd)
    plt.xlabel('People Number')
    plt.ylabel('Survive times')
    plt.title('Survive Times Odd')
    # save the figure
    plt.savefig('survive_times_odd.png')
    # clear the figure
    plt.clf()

    # draw the survive rate of each people in odd position
    survive_rate_odd = [i / SIMULATE_TIMES for i in survive_times_odd]
    plt.bar(range(1, 51), survive_rate_odd)
    plt.xlabel('People Number')
    plt.ylabel('Survive Rate')
    plt.title('Survive Rate Odd')
    # save the figure
    plt.savefig('survive_rate_odd.png')
    # clear the figure
    plt.clf()

    # draw the survive times of each people in even position
    survive_times_even = survive_times[1::2]
    plt.bar(range(1, 51), survive_times_even)
    plt.xlabel('People Number')
    plt.ylabel('Survive Times')
    plt.title('Survive Times Even')
    # save the figure
    plt.savefig('survive_times_even.png')
    # clear the figure
    plt.clf()

    # draw the survive rate of each people in even position
    survive_rate_even = [i / SIMULATE_TIMES for i in survive_times_even]
    plt.bar(range(1, 51), survive_rate_even)
    plt.xlabel('People Number')
    plt.ylabel('Survive Rate')
    plt.title('Survive Rate Even')
    # save the figure
    plt.savefig('survive_rate_even.png')
    # clear the figure
    plt.clf()
