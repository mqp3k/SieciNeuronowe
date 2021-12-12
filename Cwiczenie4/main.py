from pooling import EX_pooling
from kernel_size import EX_kernel_size
from kernel_count import EX_kernel_count
from topology import EX_topology


if __name__ == '__main__':
    EX_pooling(5, 10, 3)
    EX_kernel_size(5, 10)
    EX_kernel_count(5, 10)
    EX_topology(5, 10)