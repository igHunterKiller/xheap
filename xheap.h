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

typedef struct xheap*      xheap_t;
typedef struct xheap_pool* xheap_pool_t;
typedef void (*xheap_walker)(void* user, void* pointer, size_t size, 
                             xheap_pool_t pool, int used);

size_t       xheap_required_bytes (size_t max_allocation_count);
xheap_t      xheap_init           (void* mem, size_t max_allocation_count);
xheap_pool_t xheap_add_pool       (xheap_t heap, const void* mem, size_t bytes);
void*        xheap_remove_pool    (xheap_t heap, xheap_pool_t pool);
int32_t      xheap_validate       (xheap_t heap);
void*        xheap_malloc         (xheap_t heap, size_t bytes);
void*        xheap_realloc        (xheap_t heap, void* pointer, size_t bytes);
void*        xheap_memalign       (xheap_t heap, size_t align, size_t bytes);
void         xheap_free           (xheap_t heap, const void* pointer);
size_t       xheap_size           (const xheap_t heap, const void* pointer);

void         xheap_walk_pool      (const xheap_t heap, xheap_pool_t pool, 
                                   void* user, xheap_walker walker);
void         xheap_walk           (const xheap_t heap, 
                                   void* user, xheap_walker walker);

int xheap_unittest();

#if defined(__cplusplus)
}
#endif
#endif
