def method_one(elements, composition):
    vector = np.zeros((len(elements), len(unique_elements)))
    for i in range(len(elements)):
        for j in range(len(elements[i])):
            vector[i, np.where(unique_elements == elements[i][j])[0][0]] = composition[i][j]
    return vector

def method_two(elements, composition):
    vector = np.zeros((len(elements), MAX_LEN * 2))
    for i in range(len(elements)):
        for j in range(len(elements[i])):
            vector[i, j * 2] = element_to_index(elements[i][j])
            vector[i, j * 2 + 1] = composition[i][j]
    return vector

def method_three(elements, composition):
    params = []
    for i in range(len(elements)):
        x = np.zeros((MAX_LEN, 200))
        for j in range(len(elements[i])):
            x[j, :] = element_weights[elements[i][j]]
            x[j, int(composition[i][j] * 100)] = 1
        params.append(x)
    return np.array(params)

def method_four(elements, composition):
    params = []
    for i in range(len(elements)):
        x = np.zeros((MAX_LEN, 200))
        for j in range(len(elements[i])):
            x[j, :] = element_weights[elements[i][j]]
            x[j, :] = x[j, :] * composition[i][j]
        params.append(x)
    return np.array(params)