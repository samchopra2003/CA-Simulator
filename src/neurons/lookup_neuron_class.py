def lookup_neuron_class(neuron_id):
    """
    :param neuron_id: Must be between 1 and 17 + HIDDEN_NEURONS
    :return: neuron class
    """
    # input neurons
    if 1 <= neuron_id <= 11 :
        return 0
    # output neurons
    elif 12 <= neuron_id <= 17:
        return 2
    # hidden neurons
    return 1
