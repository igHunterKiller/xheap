# xheap
A memory allocator based on TLSF that keeps its bookkeeping outside the 
managed address space: fit for remote, write-combined or unmapped memory.

It is an implementation of the two-level segregated fit (TLSF) memory allocator.
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.469.5812&rep=rep1&type=pdf

This is a quick and dirty implementation, compiled only with Visual Studio.
