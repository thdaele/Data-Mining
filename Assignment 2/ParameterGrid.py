# Modified from https://github.com/thdaele/Information-Retrieval-Project/blob/main/src/utils.py
# to make it work with tqdm


class ParameterGrid:
    def __init__(self, grid):
        self.grid = grid.values()
        self.meaning = grid.keys()

        self.counts = [len(a) for a in self.grid]

    def __iter__(self):
        self.indices = [0 for _ in self.counts]
        self.position = 0

        self.indices[0] = -1

        return self

    def repeat(self, index):
        if index >= len(self.counts):
            raise StopIteration()

        self.indices[index] += 1
        if self.indices[index] >= self.counts[index]:
            self.indices[index] = 0
            self.repeat(index + 1)

    def __next__(self):
        self.repeat(self.position)

        return {m: g[i] for i, g, m in zip(self.indices, self.grid, self.meaning)}

    def __len__(self):
        result = 1
        for count in self.counts:
            result *= count
        return result


if __name__ == '__main__':
    for i in ParameterGrid({
        'test': [1, 2, 3],
        'other': [True, False],
        'string': ['Hello', 'String']
    }):
        print(i)
