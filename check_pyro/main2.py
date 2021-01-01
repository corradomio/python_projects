import torch
import pyro


# def weather():
#     cloudy = torch.distributions.Bernoulli(0.3).sample()
#     cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
#     mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
#     scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
#     temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()
#     return cloudy, temp.item()


loc = 0.   # mean zero
scale = 1. # unit variance

x = pyro.sample("my_sample", pyro.distributions.Normal(loc, scale))
print(x)


def weather():
    cloudy = pyro.sample('cloudy', pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 55.0, 'sunny': 75.0}[cloudy]
    scale_temp = {'cloudy': 10.0, 'sunny': 15.0}[cloudy]
    temp = pyro.sample('temp', pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()


for _ in range(3):
    print(weather())


def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream


def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)

print(geometric(0.5))


def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y

def make_normal_normal():
    mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn

print(make_normal_normal()(1.))
