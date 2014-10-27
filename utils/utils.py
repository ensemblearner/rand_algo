import numpy as np

def num_rows_each_partition(iterator):
    yield sum(1 for _ in iterator)


def indexing(splitIndex, iterator, count_each_partition):
    # count = 0
    offset = sum(count_each_partition[:splitIndex]) if splitIndex else 0
    indexed = []
    for i, e in enumerate(iterator):
        index = offset + i
        for j, ele in enumerate(e):
            indexed.append((index, j, ele))
    yield indexed


def flip(row):
    return [(e[1], e[0], e[2]) for e in row]


def extract(row):
    return [ele[2] for ele in row]


def other_iterator(itr):
    yield sum(np.outer(x, y) for x, y in itr)

def indexing_each_partition(splitIndex, iterator):
    indexing = [ (splitIndex, ele) for ele in iterator]
    yield indexing

def partition_rdd(rdd, total_count, partitions, sc):
        partitioned_rdd = sc.parallelize( rdd.take(total_count), partitions)
        indexed_partitions = partitioned_rdd.mapPartitionsWithSplit( indexing_each_partition)
        rows_each_partition = indexed_partitions.flatMap(lambda x:x).groupBy(lambda x:x[0]).map(lambda x: [ i[1] for i in x[1]])
        return rows_each_partition

def gen_synth_data(n, d):
    matrix = np.random.randn(n,d)
    return matrix.tolist(),matrix


