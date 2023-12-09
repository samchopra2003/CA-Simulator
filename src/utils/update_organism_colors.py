def update_organisms_color(species: dict, species_colors: dict):
    for name, orgs in species.items():
        color = species_colors.get(name)
        if color is not None:
            for org in orgs:
                org.color = color
