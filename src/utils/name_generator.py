import random


def generate_species_name():
    prefixes = ['bio', 'neo', 'phylo', 'proto', 'xeno', 'exo', 'quasi', 'meta', 'para']
    suffixes = ['odon', 'saur', 'raptor', 'mimus', 'morph', 'tera', 'phage', 'zoa', 'plasm', 'cyte']

    vowel_sounds = ['a', 'e', 'i', 'o', 'u', 'ae', 'ei', 'ou', 'oo', 'io']

    # Randomly choose a prefix, suffix, and vowel sound
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    vowel_sound = random.choice(vowel_sounds)

    # Combine the elements to form a species name
    species_name = f'{prefix}{vowel_sound}{suffix}'

    return species_name.capitalize()
