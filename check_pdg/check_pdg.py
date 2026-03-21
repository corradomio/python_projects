import pdg

api = pdg.connect('sqlite:///pdgall-2025-v0.2.2.sqlite')

p = api.get_particle_by_name('t')
print(api.get_particle_by_name('t'))

for decay in api.get_particle_by_name('B0').exclusive_branching_fractions():
    decay_products = [p.item.name for p in decay.decay_products]
    if 'J/psi(1S)' in decay_products:
        print(format(decay.description,'40s'), decay.display_value_text)
