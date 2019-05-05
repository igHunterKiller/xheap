/* MIT License

Copyright (c) 2019 Antony Arciuolo.

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in the 
Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/
#ifndef ouro_xheap_h
#define ouro_xheap_h
#if defined(__cplusplus)
extern "C" {
#endif

#include <stdint.h>

/*
  Adding a pool costs two allocations (max_allocation_count).

  int implies a bool type.

  xheap_t and xheap_pool_t are the same as the mem pointers used
  in their creation, so they can be cast to void* and freed with
  the backing heap that allocated the memory.

  'owns' means xheap_free on the pointer will do somethng meaningful,
  'in range' means the pointer as an integer is between some pool's
   minimum and maximum, but may not be the base pointer of an allocation.

   'name' is ignored when allow_naming is zero
*/

typedef struct xheap__*    xheap_t;
typedef struct xheap_pool* xheap_pool_t;
typedef void (*xheap_walker)(void* user, void* pointer, size_t size, 
                             const char* name, xheap_pool_t pool, int used);

/* Initialization */
size_t       xheap_calc_max_allocation_count(size_t bytes, int allow_naming);
size_t       xheap_required_bytes        (size_t max_allocation_count, int allow_naming);
xheap_t      xheap_init                  (void* mem, size_t bytes, int allow_naming);
xheap_pool_t xheap_add_pool              (xheap_t heap, const void* mem, size_t bytes, const char* name);
void*        xheap_remove_pool           (xheap_t heap, xheap_pool_t pool);

/* Inspection */
size_t       xheap_overhead              (const xheap_t heap);
size_t       xheap_allocation_count      (const xheap_t heap);
size_t       xheap_max_allocation_count  (const xheap_t heap);
size_t       xheap_allocation_highwater  (const xheap_t heap);
size_t       xheap_pool_count            (const xheap_t heap);
size_t       xheap_pool_size             (const xheap_t heap, const xheap_pool_t pool);
const char*  xheap_pool_name             (const xheap_t heap, const xheap_pool_t pool);
size_t       xheap_block_size            (const xheap_t heap, const void* pointer);
const char*  xheap_block_name            (const xheap_t heap, const void* pointer);
int          xheap_owns                  (const xheap_t heap, const void* pointer);
int          xheap_in_pool_range         (const xheap_t heap, const xheap_pool_t pool, const void* pointer);
int          xheap_in_range              (const xheap_t heap, const void* pointer);

/* Allocation/Freeing */
void*        xheap_malloc                (xheap_t heap, size_t bytes, const char* name);
void*        xheap_realloc               (xheap_t heap, void* pointer, size_t bytes, const char* name);
void*        xheap_memalign              (xheap_t heap, size_t align, size_t bytes, const char* name);
void         xheap_free                  (xheap_t heap, const void* pointer);

/* Validate returns 0 on success,  */
int32_t      xheap_validate              (const xheap_t heap);

/* Walk all blocks in a single pool */
void         xheap_walk_pool             (const xheap_t heap, xheap_pool_t pool, 
                                          void* user, xheap_walker walker);

/* Walk the first block of each pool. If used is 0, the pool can be removed. */
void         xheap_walk_pools            (const xheap_t heap, 
                                          void* user, xheap_walker walker);

/* Cycle through each pool and there cycle through each block. */
void         xheap_walk                  (const xheap_t heap, 
                                          void* user, xheap_walker walker);

int xheap_unittest();

#if defined(__cplusplus)
}
#endif
#endif
