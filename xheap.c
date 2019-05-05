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
#include "xheap.h"

#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <stdio.h>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif


/* Generic macros and helpers */

/* 64-bit v. 32-bit builds */
#if defined(__alpha__)     \
 || defined(__ia64__)      \
 || defined(__x86_64__)    \
 || defined(__powerpc64__) \
 || defined(__aarch64__)   \
 || defined(_WIN64)        \
 || defined(__LP64__)      \
 || defined(__LLP64__)
  #define xIs64Bit
#else
  #define xIs32Bit
#endif

#define xMin(x,y) ((x) < (y) ? (x) : (y))
#define xMax(x,y) ((x) > (y) ? (x) : (y))

#define xAlignUp(x, alignment)          (((uintptr_t)(x) + (((uintptr_t)alignment)-1)) & ~(((uintptr_t)alignment)-1))
#define xAlignDown(x, alignment)        ((uintptr_t)(x) & ~(((uintptr_t)alignment)-1))
#define xAligned(x, alignment)          (xAlignDown(x, alignment) == (uintptr_t)(x))
#define xPtrDiff(base, ptr)             ((ptrdiff_t)(ptr) - (ptrdiff_t)(base))


/* Simple assertion */

#define xTrace(_fmt, ...) do { printf(_fmt, ## __VA_ARGS__); } while(0)

#define xAssert(_expr, _fmt, ...) do              \
{ if (!(_expr))                                   \
  {                                               \
    xTrace("Assert: " _fmt "\n", ## __VA_ARGS__); \
    assert(_expr);                                \
  }                                               \
} while(0)


/* Wang hash */

#define wang_hash(x) ((sizeof(x) == sizeof(uint64_t)) ? wang_hash64(x) : wang_hash32((uint32_t)(x)))

inline uint32_t wang_hash32(uint32_t x)
{
  x = (x ^ 61) ^ (x >> 16);
  x = x + (x << 3);
  x = x ^ (x >> 4);
  x = x * 0x27d4eb2d;
  x = x ^ (x >> 15);
  return x;
}

inline uint64_t wang_hash64(uint64_t x)
{
  x = (~x) + (x << 21); // x = (x << 21) - x - 1;
  x = x ^ (x >> 24);
  x = (x + (x << 3)) + (x << 8); // x * 265
  x = x ^ (x >> 14);
  x = (x + (x << 2)) + (x << 4); // x * 21
  x = x ^ (x >> 28);
  x = x + (x << 31);
  return x;
}


/* Bit-twiddling functions */

inline int firstbithigh(size_t x)
{
  #if defined(_MSC_VER)
    #if defined(xIs64Bit)
      unsigned long index;
      return _BitScanReverse64(&index, x) ? index : -1;
    #else
      unsigned long index;
      return _BitScanReverse(&index, x) ? index : -1;
    #endif
  #elif defined(__clang__)
    static const uint32_t kRevCount = sizeof(size_t) * 8;
    #if defined(oIs64Bit)
      return (x ? (kRevCount - __builtin_clzll(x)) : 0) - 1;
    #else
      return (x ? (kRevCount - __builtin_clzl (x)) : 0) - 1;
    #endif
  #else
    #error unsupported compiler
  #endif
}

inline int firstbitlow(size_t x)
{
  #if defined(_MSC_VER)
    #if defined(xIs64Bit)
      unsigned long index;
      return _BitScanForward64(&index, x) ? index : -1;
    #else
      unsigned long index;
      return _BitScanForward(&index, x) ? index : -1;
    #endif
  #elif defined(__clang__)
    #if defined(oIs64Bit)
      return x ? __builtin_ctzll(x) : -1;
    #else
      return x ? __builtin_ctzl (x) : -1;
    #endif
  #else
    #error unsupported compiler
  #endif
}


/* Power of 2 utilities */

inline int log2u(size_t x)
{
  return firstbithigh(x|1);
}

inline int ispow2(size_t x)
{
  return x && !(x & (x - 1));
}

inline size_t nextpow2(size_t x)
{
  #if defined(xIs64Bit)
    #define xNProt64_(x, shift) (((uint64_t)(x))|(((uint64_t)(x)))>>(UINT64_C(shift)))
      uint64_t y = xNProt64_(xNProt64_(xNProt64_(xNProt64_(xNProt64_(x - 1, 1), 2), 4), 8), 16);
      return 1 + xNProt64_(y, 32);
    #undef xNProt64_
  #else
    #define xNProt32_(x, shift) ((x)|(x)>>(shift))
      uint32_t y = xNProt32_(xNProt32_(xNProt32_(xNProt32_(xNProt32_(x - 1, 1), 2), 4), 8), 16);
      return 1 + y;
    #undef xNProt32_
  #endif
}


/* General-purpose fixed-sized object pool. */

typedef struct pool__* pool_t;
typedef void (*pool_walk_fn)(void* user, int32_t available_index);

struct pool__
{
  uint32_t block_size : 24; /* Each allocation is of this size. */
  uint32_t align_log2 : 8;  /* First-block alignment; blocks thereafter are naturally, not explicitly aligned. */
  int32_t  capacity;        /* The maximum number of blocks allocated by this pool. */
  int32_t  used;            /* Number of blocks allocated. */
  int32_t  freelist;        /* Index to the next free block. */
};

size_t pool_calc_capacity(size_t bytes, size_t block_align, size_t block_size)
{
  size_t overhead = xAlignUp(sizeof(struct pool__), block_align);
  if (bytes < overhead)
    return 0;

  return (bytes - overhead) / block_size;
}

size_t pool_required_bytes(size_t capacity, size_t block_align, size_t block_size)
{
  return xAlignUp(sizeof(struct pool__), block_align) + capacity * block_size;
}

static char* pool_base(const pool_t pool)
{
  return (char*)xAlignUp(pool+1, ((uintptr_t)1)<<pool->align_log2);
}

int pool_owns(const pool_t pool, int32_t index)
{
  return (index >= 0 && index < pool->capacity);
}

int32_t pool_to_index(const pool_t pool, const void* pointer)
{
  int32_t index = (int32_t)(xPtrDiff(pool_base(pool), pointer) / pool->block_size);
  return pool_owns(pool, index) ? index : -1;
}

void* pool_to_pointer(const pool_t pool, int32_t index)
{
  return pool_owns(pool, index) 
    ? (void*)(pool_base(pool) + index * pool->block_size) 
    : NULL;
}

int32_t pool_alloc(pool_t pool)
{
  int32_t index = pool->freelist;
  if (index != -1)
  {
    int32_t* ptr = (int32_t*)pool_to_pointer(pool, index);
    pool->freelist = *ptr;
    pool->used++;
    #if DEBUG_USE_SHRED_VALUES
      memset(ptr, DEBUG_ALLOC_SHRED, pool->block_size);
    #endif
  }
  return index;
}

void pool_free(pool_t pool, int32_t index)
{
  if (pool_owns(pool, index))
  {
    int32_t* ptr = (int32_t*)pool_to_pointer(pool, index);
    *ptr = pool->freelist;
    #if DEBUG_USE_SHRED_VALUES
      memset(ptr+1, DEBUG_FREE_SHRED, pool->block_size - sizeof(int32_t));
    #endif
    pool->freelist = index;
    pool->used--;
  }
}

size_t pool_capacity(const pool_t pool)
{
  return pool->capacity;
}

size_t pool_used(const pool_t pool)
{
  return pool->used;
}

pool_t pool_init(void* mem, size_t bytes, size_t block_align, size_t block_size)
{
  const size_t capacity = pool_calc_capacity(bytes, block_align, block_size);

  pool_t pool      = (pool_t)mem;
  pool->block_size  = (uint32_t)block_size;
  pool->align_log2  = (uint32_t)log2u(block_align);
  pool->capacity    = (uint32_t)capacity;
  pool->used        = 0;
  pool->freelist    = 0;
  ptrdiff_t pad     = xPtrDiff(pool+1, pool_base(pool));
  memset(pool+1, 0, pad);

  #if DEBUG_USE_SHRED_VALUES
    memset(pool_base(pool), DEBUG_INIT_SHRED, block_size * pool_capacity(pool));
  #endif

  const int32_t last = pool->capacity - 1;
  for (int32_t i = 0; i < last; i++)
    *(int32_t*)pool_to_pointer(pool, i) = i + 1;
  *(int32_t*)pool_to_pointer(pool, last) = -1;

  return (!mem 
    || pool->block_size != block_size 
    || (((uintptr_t)1)<<pool->align_log2) != block_align)
      ? NULL
      : pool;
}

void pool_walk_freelist(const pool_t pool, void* user, pool_walk_fn walker)
{
  int32_t index = pool->freelist;
  while (index != -1)
  {
    walker(user, index);
    index = *(int32_t*)pool_to_pointer(pool, index);
  }
}


/* General-purpose linear probe, robin hood, backwards-shift-delete hashmap. */

typedef struct hashmap__* hashmap_t;
typedef void (*hashmap_walk_fn)(void* user, uint64_t hash, int32_t value);

struct hashmap__
{
  uint32_t fibshift; /* https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/ */
  uint32_t wrapmask; /* Replaces modulo of capacity with a mask. */
  uint32_t count;    /* Number of valid entries. */
  uint32_t padA;     /* Unused. Ensure natural alignment to sizeof(uint64_t). */
};

static uint64_t* hashmap_keys(hashmap_t map)
{
  return (uint64_t*)(map + 1);
}

int hashmap_capacity(const hashmap_t map)
{
  return map ? (map->wrapmask + 1) : 0;
}

static int* hashmap_vals(hashmap_t map)
{
  return (int*)(hashmap_keys(map) + hashmap_capacity(map));
}

static uint32_t hashmap_to_slot(const hashmap_t map, uint64_t hash)
{
  /* Fibonacci (multiplicative) hashing. */
  uint64_t h = hash ^ (hash >> map->fibshift);
  return (uint32_t)((11400714819323198485llu * h) >> map->fibshift);
}

static uint32_t hashmap_probe_dist(const hashmap_t map, uint64_t hash, uint32_t slot)
{
  return (slot - hashmap_to_slot(map, hash)) & map->wrapmask;
}

size_t hashmap_calc_capacity(size_t bytes)
{
  size_t overhead = sizeof(struct hashmap__);
  if (bytes < overhead)
    return 0;

  size_t entry_bytes = bytes - overhead;
  size_t entry_count = entry_bytes / (sizeof(uint64_t) + sizeof(int32_t));
  
  if (!ispow2(entry_count))
  {
    entry_count = nextpow2(entry_count) >> 1;
  }

  return entry_count;
}

size_t hashmap_required_bytes(size_t entry_capacity)
{
  const size_t nentries = nextpow2(entry_capacity);
  return sizeof(struct hashmap__) + nentries * (sizeof(uint64_t) + sizeof(int));
}

hashmap_t hashmap_init(void* mem, size_t bytes)
{
  const size_t capacity = hashmap_calc_capacity(bytes);
  memset(mem, 0, bytes);

  hashmap_t map = (hashmap_t)mem;
  map->wrapmask = (uint32_t)capacity - 1;
  map->fibshift = (uint32_t)((sizeof(uint64_t)*8) - log2u(capacity));
  map->count    = 0;
  map->padA     = 0;

  return map;
}

int hashmap_count(const hashmap_t map)
{
  return map ? map->count : 0;
}

int hashmap_find_slot(const hashmap_t map, uint64_t hash)
{
  /* Ensure hash is never actually zero. */
  hash |= hash == 0;

  uint64_t*      keys = hashmap_keys(map);
  uint32_t       slot = hashmap_to_slot(map, hash);
  const uint32_t wrap = map->wrapmask;

  uint32_t dist = 0;
  while (1)
  {
    if (keys[slot] == 0)
      return -1;
    else if (dist > hashmap_probe_dist(map, keys[slot], slot))
      return -1;
    else if (keys[slot] == hash)
      return slot;
    
    dist++;
    slot = (slot + 1) & wrap;
  }
}

int hashmap_get(const hashmap_t map, uint64_t hash)
{
  const int slot = hashmap_find_slot(map, hash);
  return slot == -1 ? -1 : hashmap_vals(map)[slot];
}

int hashmap_set(hashmap_t map, uint64_t hash, int value)
{
  /* Only positive values, but int for the convenience of invalid == -1 */
  if (value < 0)
  {
    return -1;
  }

  /* Ensure hash is never actually zero. */
  hash |= hash == 0;

  uint64_t*      keys = hashmap_keys(map);
  int*           vals = hashmap_vals(map);
  uint32_t       slot = hashmap_to_slot(map, hash);
  const uint32_t wrap = map->wrapmask;
  const uint32_t stop = slot;

  /* Probe (Robin Hood) */
  int dist = 0;

  const int allow_rebalance = hashmap_count(map) < hashmap_capacity(map);

  int v = value;
  do
  {
    /* Replace any existing. */
    if (keys[slot] == hash)
    {
      int old_value = vals[slot];
      vals[slot] = v;
      return old_value;
    }
  
    /* Assign first empty. */
    if (keys[slot] == 0)
    {
      keys[slot] = hash;
      vals[slot] = v;
      map->count++;
      return value;
    }
    
    /* If slot not empty, seek to rebalance. */
    if (allow_rebalance)
    {
      int cur_dist = hashmap_probe_dist(map, keys[slot], slot);
      if (cur_dist < dist)
      {
        uint64_t tmp_hash = keys[slot];
        keys[slot] = hash;
        hash = tmp_hash;
    
        int tmp_val = vals[slot];
        vals[slot] = v;
        v = tmp_val;
        dist = cur_dist;
      }
      dist++;
    }
   
    slot = (slot + 1) & wrap;
  
  } while (slot != stop);

  return -1;
}

int hashmap_remove(hashmap_t map, uint64_t hash)
{
  int slot = hashmap_find_slot(map, hash);
  if (slot != -1)
  {
    /* Backwards shift deletion */
    uint64_t*      keys = hashmap_keys(map);
    int*           vals = hashmap_vals(map);
    const uint32_t wrap = map->wrapmask;
    int            val  = vals[slot];

    int prev = slot;
    slot = (slot + 1) & wrap;
    while (keys[slot] != 0 && hashmap_probe_dist(map, keys[slot], slot) != 0)
    {
      keys[prev] = keys[slot];
      vals[prev] = vals[slot];
      prev = slot;
      slot = (slot + 1) & wrap;
    }
    keys[prev] = 0;
    map->count--;
    return val;
  }

  return -1;
}

void hashmap_walk(const hashmap_t map, void* user, hashmap_walk_fn walker)
{
  const uint64_t* keys     = hashmap_keys(map);
  const int*      vals     = hashmap_vals(map);
  const uint32_t  capacity = hashmap_capacity(map);

  for (uint32_t i = 0; i < capacity; i++)
  {
    if (keys[i] != 0)
    {
      walker(user, keys[i], vals[i]);
    }
  }
}


/* xheap-specific code */

enum Constants
{
  kSLIndexCountLog2   = 5,
  #if defined(xIs64Bit)
    kMinAlignmentLog2 = 3,
    kMaxFLIndex       = 32,
  #else
    kMinAlignmentLog2 = 2,
    kMaxFLIndex       = 30,
  #endif
  kMinAlignment       = 1 << kMinAlignmentLog2,
  kSLIndexCount       = 1 << kSLIndexCountLog2,
  kFLIndexShift       = kSLIndexCountLog2 + kMinAlignmentLog2,
  kFLIndexCount       = kMaxFLIndex - kFLIndexShift + 1,
  kSmallBlockSize     = 1 << kFLIndexShift,

};

static const size_t kMinBlockSize = kMinAlignment;
static const size_t kMaxBlockSize = ((size_t)1) << kMaxFLIndex;


/* Ensure assumptions hold. */
static_assert((sizeof(uint32_t) * 8) >= kSLIndexCount, "");
static_assert(kMinAlignment == (kSmallBlockSize / kSLIndexCount), "");

/* Validation macro */
#define xCheck(expr, fmt, ...) do { if (!(expr)) { nerrors++; xTrace(fmt, ## __VA_ARGS__); } } while(0)


typedef struct block_t
{
  /* Blocks are stored outside the managed heap. */

  /* Address range */
  uintptr_t offset;
  size_t    size;

  /* Contiguous block linked-list */
  int32_t contiguous_prev;
  int32_t contiguous_next;

  /* Freelist linked-list */
  int32_t free_prev;
  int32_t free_next : 31;
  int32_t is_free   :  1;

} block_t;

typedef struct pool_block_t
{
  /* Total pool address range */
  uintptr_t offset;
  size_t    size;

  /* Indexes the blocks in the freelist. */
  int32_t first_block;
  int32_t last_block;

  /* Poollist linked-list */
  int32_t pool_prev;
  int32_t pool_next;

} pool_block_t;

static_assert(sizeof(block_t) == sizeof(pool_block_t), "block_t and pool_block_t alias the same memory, so must be the same size");

typedef struct freelist_t
{
  /* TLSF bitfields and indices into available blocks. */

  uint32_t fl_bitmap;
  uint32_t sl_bitmap[kFLIndexCount];
  int32_t  freelists[kFLIndexCount][kSLIndexCount];
} freelist_t;

struct xheap__
{
  freelist_t   freelist;        /* TLSF details for quick freelist location. */
  pool_t       blocks;          /* Blocks are allocated and deallocated from here. */
  hashmap_t    lookup;          /* Maps pointer to block index. */
  int32_t      pools;           /* Head of a list of pools (by index, -1 is terminator). */
  int32_t      block_highwater; /* Highest count of used blocks. */
  const char** block_names;     /* Indexed by block_index/pool_block_index */
  size_t       xheap_bytes;     /* Size of working memory used to initialize the xheap. */
};

static_assert(xAligned(sizeof(struct xheap__), sizeof(void*)), "No explicit size alignment is done on struct, so ensure it's natural alignment is reasonable.");


/* TLSF core algorithm */

static void tlsf_init(freelist_t* freelist)
{
  freelist->fl_bitmap = 0;
	for (int32_t i = 0; i < kFLIndexCount; i++)
	{
    freelist->sl_bitmap[i] = 0;
		for (int32_t j = 0; j < kSLIndexCount; j++)
      freelist->freelists[i][j] = -1;
	}
}

static void tlsf_insert(size_t size, int32_t* fl, int32_t* sl)
{
  int32_t f, s;

	if (size < kSmallBlockSize)
	{
		f = 0;
		s = ((int32_t)size) / (kSmallBlockSize / kSLIndexCount);
	}
	else
	{
		f = firstbithigh(size);
		s = (int32_t)(size >> (f - kSLIndexCountLog2)) ^ (1 << kSLIndexCountLog2);
		f -= (kFLIndexShift - 1);
	}
  *fl = f;
  *sl = s;
}

static void tlsf_search(size_t size, int32_t* fl, int32_t* sl)
{
	if (size >= kSmallBlockSize)
		size += (1 << (firstbithigh(size) - kSLIndexCountLog2)) - 1;
	tlsf_insert(size, fl, sl);
}

static int32_t tlsf_search_suitable_block(const xheap_t heap, int32_t* fli, int32_t* sli)
{
  /* Do not read off the end of FL bits */
  if (*fli >= kFLIndexCount)
    return -1;

  int32_t fl = *fli;
	int32_t sl = *sli;

  const freelist_t* freelist = &heap->freelist;
	uint32_t sl_map = freelist->sl_bitmap[fl] & (~0U << sl);
	if (!sl_map)
	{
		const uint32_t fl_map = freelist->fl_bitmap & (~0U << (fl + 1));
		if (!fl_map)
		{
			/* Free blocks exhausted, out of memory. */
			return -1;
		}

		fl = firstbitlow(fl_map);
		*fli = fl;
		sl_map = freelist->sl_bitmap[fl];
	}
	
  sl = firstbitlow(sl_map);
	*sli = sl;

	return freelist->freelists[fl][sl];
}

static void tlsf_mark_free(xheap_t heap, int32_t fl, int32_t sl)
{
  heap->freelist.fl_bitmap     |= (1 << fl);
  heap->freelist.sl_bitmap[fl] |= (1 << sl);
}

static void tlsf_mark_removed_from_freelist(xheap_t heap, int32_t fl, int32_t sl)
{
  freelist_t* freelist = &heap->freelist;

  /* If the freelist is now empty, update bitfields. */
	if (freelist->freelists[fl][sl] == -1)
	{
    freelist->sl_bitmap[fl] &= ~(1 << sl);

		/* If the second bitmap is now empty, clear the fl bitmap. */
		if (!freelist->sl_bitmap[fl])
		{
      freelist->fl_bitmap &= ~(1 << fl);
		}
	}
}

static size_t tlsf_adjust_size(size_t size, size_t align)
{
  /*  kMaxBlockSize is the last size addressable by TLSF's SL, 
      so don't overstep that bit capacity.
  */
	size_t aligned = size > 0 ? xAlignUp(size, align) : 0;
  if (aligned >= kMaxBlockSize) 
		aligned = 0;
	return aligned;
}


/* Hashing */

static size_t hash_pointer(const void* pointer)
{
  return wang_hash((uintptr_t)pointer);
}

static size_t hash_pool(const xheap_pool_t pool)
{
  /* Differentiate between pool bookkeeping and a user allocation block. */
  return wang_hash((uintptr_t)pool | 1);
}


/* Block mapping and traversal */

static block_t* block_from_index(const xheap_t heap, int32_t block_index)
{
  block_t* block = (block_t*)pool_to_pointer(heap->blocks, block_index);
  return block;
}

static block_t* block_from_pointer(const xheap_t heap, const void* pointer)
{
  uint64_t hash       = hash_pointer(pointer);
  int32_t block_index = hashmap_get(heap->lookup, hash);
  return block_from_index(heap, block_index);
}

static int32_t block_to_index(const xheap_t heap, const block_t* block)
{
  return pool_to_index(heap->blocks, block);
}

static void* block_to_pointer(const block_t* block)
{
  return (void*)block->offset;
}

static const char* block_name(const xheap_t heap, int32_t block_index)
{
  return heap->block_names ? heap->block_names[block_index] : "(naming not supported)";
}


/* Freelist management */

static void freelist_remove_specific(xheap_t heap, int32_t* freelist, block_t* block)
{
  const int32_t next_index = block->free_next;

  if (next_index != -1)
  {
	  block_t* next = block_from_index(heap, next_index);
    next->free_prev = block->free_prev;
    block->free_next = -1;
  }
  
  if (block->free_prev == -1)
  {
    /* If block is the head of the freelist. */
    *freelist = next_index;
  }
  else
  {
    /* If Otherwise repatch connectivity around block. */
    block_t* prev = block_from_index(heap, block->free_prev);
	  prev->free_next = block->free_next;
  }

  if (block->free_prev == -1)
  {
    *freelist = next_index;
  }
  else
  {
    block->free_prev = -1;
  }
}

static void freelist_remove_flsl(xheap_t heap, block_t* block, int32_t fl, int32_t sl)
{
  freelist_t* freelist = &heap->freelist;
  freelist_remove_specific(heap, &freelist->freelists[fl][sl], block);
  tlsf_mark_removed_from_freelist(heap, fl, sl);
}

static void freelist_remove(xheap_t heap, block_t* block)
{
	int32_t fl = 0, sl = 0;
	tlsf_insert(block->size, &fl, &sl);
	freelist_remove_flsl(heap, block, fl, sl);
}

static void freelist_insert(xheap_t heap, block_t* block)
{
	int32_t fl, sl;
	tlsf_insert(block->size, &fl, &sl);

  int32_t* freelist = &heap->freelist.freelists[fl][sl];

  /* Init block to be pointing to what's in the head now, and thus no previous. */
  block->free_next     = *freelist;
  block->free_prev     = -1;
  int32_t block_index  = block_to_index(heap, block);

  /* If there was a head, update it to point to block (the new head). */
  if (*freelist != -1)
  {
    block_t* head = block_from_index(heap, *freelist);
    head->free_prev = block_index;
  }

  *freelist = block_index;

  tlsf_mark_free(heap, fl, sl);
}

static block_t* freelist_alloc(xheap_t heap, size_t size, int32_t* out_block_index)
{
  xAssert(size, "zero size is not supported");

  block_t* block   = NULL;
  *out_block_index = -1;
	int32_t fl = 0, sl = 0;
	tlsf_search(size, &fl, &sl);
	*out_block_index = tlsf_search_suitable_block(heap, &fl, &sl);
  if (*out_block_index != -1)
  {
    block = block_from_index(heap, *out_block_index);
    freelist_remove_flsl(heap, block, fl, sl);
  }
	return block;
}


/* Block split/merge */

static int32_t block_alloc(xheap_t heap)
{
  int32_t block_index = pool_alloc(heap->blocks);
  heap->block_highwater = xMax(heap->block_highwater, (int32_t)pool_used(heap->blocks));
  return block_index;
}

static void block_free(xheap_t heap, int32_t block_index)
{
  pool_free(heap->blocks, block_index);
  if (heap->block_names)
  {
    heap->block_names[block_index] = "(unallocated)";
  }
}

static int32_t block_can_split(block_t* block, size_t size)
{
	return block->size > size;
}

static block_t* block_split_to_next(xheap_t heap, block_t* block, size_t goal_block_size)
{
  int32_t new_next_block_index = block_alloc(heap);
  if (new_next_block_index == -1)
    return NULL;

  block_t* new_next_block = block_from_index(heap, new_next_block_index);

  new_next_block->offset  = block->offset + goal_block_size;
  new_next_block->size    = block->size - goal_block_size;
  xAssert(xAligned(new_next_block->size, kMinAlignment), "misaligned size, status flags will be affected");
  
  new_next_block->contiguous_next = block->contiguous_next;
  new_next_block->contiguous_prev = block_to_index(heap, block);
  new_next_block->free_next       = -1;
  new_next_block->free_prev       = -1;
  new_next_block->is_free         = -1;

  if (new_next_block->contiguous_next != -1)
  {
    block_t* next_next = block_from_index(heap, new_next_block->contiguous_next);
    next_next->contiguous_prev = new_next_block_index;
  }

  block->size = goal_block_size;
  block->contiguous_next = new_next_block_index;

  return new_next_block;
}

static block_t* block_split_to_prev(xheap_t heap, block_t* block, size_t goal_block_size)
{
  int32_t new_prev_block_index = block_alloc(heap);
  if (new_prev_block_index == -1)
    return NULL;

  block_t* new_prev_block = block_from_index(heap, new_prev_block_index);

  new_prev_block->offset  = block->offset;
  new_prev_block->size    = block->size - goal_block_size;
  xAssert(xAligned(new_prev_block->size, kMinAlignment), "misaligned size, status flags will be affected");
  
  new_prev_block->contiguous_next = block_to_index(heap, block);
  new_prev_block->contiguous_prev = block->contiguous_prev;
  new_prev_block->free_next       = -1;
  new_prev_block->free_prev       = -1;
  new_prev_block->is_free         = -1;

  if (new_prev_block->contiguous_prev != -1)
  {
    block_t* prev_prev = block_from_index(heap, new_prev_block->contiguous_prev);
    prev_prev->contiguous_next = new_prev_block_index;
  }

  block->offset += new_prev_block->size;
  block->size = goal_block_size;
  block->contiguous_prev = new_prev_block_index;

  return new_prev_block;
}

static void block_merge_prev(xheap_t heap, block_t* block)
{
  if (block->contiguous_prev == -1)
    return;

  int32_t prev_block_index = block->contiguous_prev;

  block_t* prev = block_from_index(heap, prev_block_index);
  if (!prev->is_free)
    return;

  freelist_remove(heap, prev);

  xAssert((prev->offset + prev->size) == block->offset, "block_merge_prev - blocks aren't contiguous");
  block->contiguous_prev = prev->contiguous_prev;

  if (prev->contiguous_prev != -1)
  {
    block_t* prev_prev = block_from_index(heap, prev->contiguous_prev);
    prev_prev->contiguous_next = block_to_index(heap, block);
  }

  block->offset = prev->offset;
  block->size  += prev->size;

  xAssert(prev->free_next == -1, "");
  xAssert(prev->free_prev == -1, "");

  block_free(heap, prev_block_index);
}

static void block_merge_next(xheap_t heap, block_t* block)
{
  if (block->contiguous_next == -1)
    return;

  int32_t next_block_index = block->contiguous_next;

  block_t* next = block_from_index(heap, next_block_index);
  if (!next->is_free)
    return;

  freelist_remove(heap, next);

  xAssert((block->offset + block->size) == next->offset, "block_merge_next - blocks aren't contiguous");
  block->contiguous_next = next->contiguous_next;

  if (next->contiguous_next != -1)
  {
    block_t* next_next = block_from_index(heap, next->contiguous_next);
    next_next->contiguous_prev = block_to_index(heap, block);
  }

  block->size += next->size;

  xAssert(next->free_next == -1, "");
  xAssert(next->free_prev == -1, "");

  block_free(heap, next_block_index);
}

static void* block_finalize_as_used(xheap_t heap, int32_t block_index, block_t* block, size_t size, const char* name)
{
	void* pointer = 0;
	if (block)
	{
		xAssert(size, "zero size block found (not supported)");
	  xAssert(block->is_free, "allocated a block already in use");

    /*  The block is a best-fit match, so recover excess memory to 
        form an exact-fit block.
    */
    if (block_can_split(block, size))
	  {
		  block_t* new_trimmed = block_split_to_next(heap, block, size);
		  freelist_insert(heap, new_trimmed);
	  }
		
	  block->is_free = 0;
    
    pointer = block_to_pointer(block);

    uint64_t hash = hash_pointer(pointer);
    hashmap_set(heap->lookup, hash, block_index);

    if (heap->block_names)
    {
      heap->block_names[block_index] = name ? name : "(unnamed block)";
    }
	}
	return pointer;
}


/* XHeap Pool */

static pool_block_t* pool_block_from_index(const xheap_t heap, int32_t pool_block_index)
{
  pool_block_t* pool_block = (pool_block_t*)pool_to_pointer(heap->blocks, pool_block_index);
  return pool_block;
}

static void poollist_remove(xheap_t heap, pool_block_t* pool_block)
{
  if (pool_block->pool_prev == -1)
  {
    heap->pools = pool_block->pool_next;
  }
  else
  {
    pool_block_t* prev_pool_block = pool_block_from_index(heap, pool_block->pool_prev);
    prev_pool_block->pool_next = pool_block->pool_next;
  }

  if (pool_block->pool_next != -1)
  {
    pool_block_t* next_pool_block = pool_block_from_index(heap, pool_block->pool_next);
    next_pool_block->pool_prev = pool_block->pool_prev;
  }
}

static int32_t pool_validate(const xheap_t heap, const pool_block_t* pool_block)
{
  int32_t nerrors = 0;

  if (pool_block)
  {
    const xheap_pool_t pool = (const xheap_pool_t)pool_block->offset;

    /* First user block in the pool. */
    uint64_t hash       = hash_pool(pool);
    int32_t  prev_index = hashmap_get(heap->lookup, hash);
    
    block_t* prev = block_from_index(heap, prev_index);
    int32_t block_index = prev->contiguous_next;
    while (block_index != -1)
    {
      block_t* block = block_from_index(heap, block_index);
      if (block)
      {
        xCheck((prev->offset + prev->size) == block->offset, "Blocks %p(%d) and %p(%d) are not contiguous", prev, prev_index, block, block_index);

        prev_index = block_index;
        prev = block;
        block_index = block->contiguous_next;
      }
      else
      {
        nerrors++;
        block_index = -1;
      }
    }
  }

	return nerrors;
}

static pool_block_t* pool_to_pool_block(const xheap_t heap, const xheap_pool_t pool)
{
  uint64_t pool_hash        = hash_pool(pool);
  int32_t  pool_block_index = hashmap_get(heap->lookup, pool_hash);
  if (pool_block_index == -1)
    return NULL;

  pool_block_t* pool_block = pool_block_from_index(heap, pool_block_index);

  xAssert(pool_block && pool_block->offset == (const uintptr_t)pool, 
    "Mismatched pool lookup: pool %p, pool_hash %" PRIu64 ", pool_block_index %d, pool_block %p",
    pool, pool_hash, pool_block_index, pool_block);

  return pool_block;
}

static int32_t pool_is_single_block(const xheap_t heap, const pool_block_t* pool_block)
{
  const block_t* first_block = block_from_index(heap, pool_block->first_block);
  return first_block->contiguous_next == pool_block->last_block;
}


/* Heap */

static int32_t heap_validate_tlsf(const xheap_t heap)
{
  /* Walk the TSLF control bits and ensure the referenced blocks are in
     an appropriate state.
  */
  const freelist_t* freelist = &heap->freelist;
  int32_t nerrors = 0;

	/* Check that the free lists and bitmaps are accurate. */
	for (int32_t i = 0; i < kFLIndexCount; ++i)
	{
		for (int32_t j = 0; j < kSLIndexCount; ++j)
		{
			const int32_t  fl_map      = freelist->fl_bitmap & (1 << i);
			const int32_t  sl_list     = freelist->sl_bitmap[i];
			const int32_t  sl_map      = sl_list & (1 << j);
            int32_t  block_index = freelist->freelists[i][j];
			const block_t* block       = block_from_index(heap, block_index);

			if (!fl_map)
			{
				xCheck(!sl_map, "expected second-level bit to be as empty as first-level bit");
			}

			if (!sl_map)
			{
				xCheck(block_index == -1, "expected freelist to be empty");
				continue;
			}

			xCheck(sl_list, "expected freelist to be validly flagged in second-level");
			xCheck(block_index != -1, "expected a non-empty freelist");
			while (block_index != -1)
			{
				xCheck(block->is_free, "expected block %p(%d) to be free", block, block_index);
				
				xCheck(!block_from_index(heap, block->contiguous_next)->is_free,
          "block %p(%d) and next block %p(%d) should have merged because both are free", 
          block, block_index, block_from_index(heap, block->contiguous_next), block_to_index(heap, block_from_index(heap, block->contiguous_next)));

				int32_t fl = 0, sl = 0;
				tlsf_insert(block->size, &fl, &sl);

        xCheck(fl == i && sl == j,
          "block %p(%d) size (%zu) indexed in wrong list FL=%d SL=%d",
          block, block_index, block->size, fl, sl);

        block_index = block->free_next;
        block = block_from_index(heap, block_index);
			}
		}
	}

  return nerrors;
}


/* Public interface */

size_t xheap_calc_max_allocation_count(size_t bytes, int allow_naming)
{
  /* Hashmap rounds to nextpow2, so this isn't straightforward... */

  size_t xheap_overhead   = xheap_required_bytes(0, allow_naming);
  size_t hashmap_overhead = hashmap_required_bytes(0);

  size_t working_bytes = bytes - xheap_overhead + hashmap_overhead;

  size_t guess_count = hashmap_calc_capacity(working_bytes);
  size_t guess_bytes = xheap_required_bytes(guess_count, allow_naming);

  /* Shrink fast at about the rate hashmap does */
  size_t guess_count_max = 0;
  while (guess_bytes > bytes)
  {
    guess_count_max = guess_count;
    guess_count = nextpow2(guess_count) >> 1;
    guess_bytes = xheap_required_bytes(guess_count, allow_naming);
  }

  /* Binary search for the best fit. */
  size_t bs_left  = guess_count; /* Lowest so far. */
  size_t bs_right = guess_count_max; /* Closest that still failed above loop. */

  while (bs_left <= bs_right)
  {
    guess_count = (bs_left + bs_right) / 2;
    guess_bytes = xheap_required_bytes(guess_count, allow_naming);

    if (guess_bytes == bytes)
    {
      break;
    }
    else if (guess_bytes < bytes)
    {
      /* Is this the largest guess that isn't over? */
      guess_bytes = xheap_required_bytes(guess_count+1, allow_naming);
      if (guess_bytes > bytes)
        break;

      bs_left = guess_count + 1;
    }
    else
    {
      bs_right = guess_count - 1;
    }
  }

  if (guess_bytes > bytes)
    guess_count;

  return guess_count;
}

size_t xheap_required_bytes(size_t max_allocation_count, int allow_naming)
{
  size_t pool_bytes   = pool_required_bytes(max_allocation_count, sizeof(void*), sizeof(block_t));
  size_t lookup_bytes = hashmap_required_bytes(max_allocation_count);
  size_t name_bytes   = max_allocation_count * sizeof(const char*);
  return sizeof(struct xheap__)
    + pool_bytes 
    + lookup_bytes
    + (allow_naming ? name_bytes : 0);
}

xheap_t xheap_init(void* mem, size_t bytes, int allow_naming)
{
  if (!mem || bytes < sizeof(struct xheap__))
    return NULL;

  if (!xAligned(mem, kMinAlignment))
		return NULL;

  xheap_t heap = (xheap_t)mem;

  tlsf_init(&heap->freelist);

  const size_t max_allocation_count = xheap_calc_max_allocation_count(bytes, allow_naming);
  xAssert(xheap_required_bytes(max_allocation_count, allow_naming) <= bytes, "error calculating max_allocation_count");
  xAssert(xheap_required_bytes(max_allocation_count+1, allow_naming) > bytes, "inefficient calculation of max_allocation_count");

  size_t lookup_bytes = hashmap_required_bytes(max_allocation_count);
  size_t blocks_bytes = pool_required_bytes   (max_allocation_count, sizeof(void*), sizeof(block_t));

  void* lookup_mem = heap + 1;
  void* blocks_mem = (char*)lookup_mem + lookup_bytes;
  void* naming_mem = (char*)blocks_mem + blocks_bytes;

  xAssert(xAligned(naming_mem, sizeof(const char*)), "misalignment");

  heap->lookup          = hashmap_init(lookup_mem, lookup_bytes);
  heap->blocks          = pool_init(blocks_mem, blocks_bytes, sizeof(void*), sizeof(block_t));
  heap->pools           = -1;
  heap->block_highwater = 0;
  heap->block_names     = allow_naming ? (const char**)naming_mem : (const char**)NULL;
  heap->xheap_bytes     = bytes;

  if (heap->block_names)
  {
    for (int32_t i = 0; i < max_allocation_count; i++)
    {
      heap->block_names[i] = "(unallocated)";
    }
  }

	return heap;
}

size_t xheap_overhead(const xheap_t heap)
{
  return heap ? heap->xheap_bytes : 0;
}

size_t xheap_allocation_count(const xheap_t heap)
{
  return heap ? pool_used(heap->blocks) : 0;
}

size_t xheap_max_allocation_count(const xheap_t heap)
{
  return heap ? pool_capacity(heap->blocks) : 0;
}

size_t xheap_allocation_highwater(const xheap_t heap)
{
  return heap ? heap->block_highwater : 0;
}

xheap_pool_t xheap_add_pool(xheap_t heap, const void* mem, size_t bytes, const char* name)
{
  xAssert(xAligned(mem, kMinAlignment), "pools must be %d-byte aligned", kMinAlignment);

	xheap_pool_t pool       = (xheap_pool_t)mem;
  size_t       pool_bytes = xAlignDown(bytes, kMinAlignment);
  
  xAssert(pool_bytes >= kMinBlockSize && pool_bytes <= kMaxBlockSize, 
    "pools must be between %zu and %zu bytes", kMinBlockSize, kMaxBlockSize);

  /* Allocate blocks: 1 special pool block and begin/end blocks. */

  int32_t pool_block_index = block_alloc(heap);
  if (pool_block_index)
  {
    return 0;
  }

  int32_t block_index = block_alloc(heap);
  if (block_index == -1)
  {
    block_free(heap, pool_block_index);
    return 0;
  }

  int32_t next_index = block_alloc(heap);
  if (next_index == -1)
  {
    block_free(heap, block_index);
    block_free(heap, pool_block_index);
    return 0;
  }

  /* Initialize the base free block. */
  block_t* block         = block_from_index(heap, block_index);
  block->offset          = 0x1800;//(uintptr_t)mem;
  block->contiguous_prev = -1;
  block->contiguous_next = next_index;
  block->size            = pool_bytes;
  block->free_prev       = -1;
  block->free_next       = -1;
	block->is_free         = -1;

	
  /* Initialize a sentinel block to keep pools non-contiguous. */
  block_t* next         = block_from_index(heap, next_index);
  next->offset          = block->offset + block->size;
  next->size            = 0;
  next->contiguous_prev = block_index;
  next->contiguous_next = -1;
  next->free_prev       = -1;
  next->free_next       = -1;
	next->is_free         = 0;


  /* Initialize pool block */
  pool_block_t* pool_block    = pool_block_from_index(heap, pool_block_index);
  pool_block->offset          = (uintptr_t)mem;
  pool_block->size            = pool_bytes;
  pool_block->first_block     = block_index;
  pool_block->last_block      = next_index;
  pool_block->pool_prev       = -1;
  pool_block->pool_next       = heap->pools;


  /* Register with heap. */
  heap->pools = pool_block_index;

  uint64_t pool_hash = hash_pool(pool);
  hashmap_set(heap->lookup, pool_hash, pool_block_index);

  if (heap->block_names)
  {
    heap->block_names[pool_block_index] = name ? name : "(unnamed pool)";
  }


  /* Pool is available as a new free block. */
	freelist_insert(heap, block);

	return pool;
}

void* xheap_remove_pool(xheap_t heap, xheap_pool_t pool)
{
  uint64_t pool_hash        = hash_pool(pool);
  int32_t  pool_block_index = hashmap_remove(heap->lookup, pool_hash);
  if (pool_block_index == -1)
    return 0;

  /* Remove from all-pools linked list. */
  pool_block_t* pool_block = pool_block_from_index(heap, pool_block_index);

  /* Find user blocks and ensure the whole pool is contiguously free. */
  xAssert((uintptr_t)pool == pool_block->offset, "pool to pool block reference is mismatched");

  int32_t block_index = pool_block->first_block;
  int32_t next_index  = pool_block->last_block;

  block_t* block = block_from_index(heap, block_index);
  if (!block->is_free || next_index != block->contiguous_next)
    return 0;

  block_t* next = block_from_index(heap, next_index);

  /* Sentinel block is zero-sized. */
  if (next->size != 0)
    return 0;

  xAssert(!next->is_free, "sentinel block should be indicated as used (not-free)");
  xAssert(block->size == pool_block->size, "pool and pool block sizes mismatch (%zu != %zu)", block->size, pool_block->size);

  /* Remove all entries. */
  poollist_remove(heap, pool_block);
  freelist_remove(heap, block);

  block_free(heap, block_index);
  block_free(heap, next_index);
  block_free(heap, pool_block_index);

  return (void*)pool;
}

size_t xheap_pool_count(const xheap_t heap)
{
  size_t pool_count = 0;

  int32_t pool_block_index = heap->pools;
  while (pool_block_index != -1)
  {
    pool_count++;

    pool_block_t* pool_block = pool_block_from_index(heap, pool_block_index);
    pool_block_index = pool_block->pool_next;
  }
  
  return pool_count;
}

size_t xheap_block_size(const xheap_t heap, const void* pointer)
{
  block_t* block = block_from_pointer(heap, pointer);
  if (!block)
    return 0;
  xAssert((uintptr_t)pointer == block->offset, "invalid pointer %p mapping to block %p(%d)", pointer, block, block_to_index(heap, block));
  return block->size;
}

size_t xheap_pool_size(const xheap_t heap, const xheap_pool_t pool)
{
  pool_block_t* pool_block = pool_to_pool_block(heap, pool);
  if (!pool_block)
    return 0;

  return pool_block->size;
}

const char* xheap_pool_name(const xheap_t heap, const xheap_pool_t pool)
{
  const char* name = "(not tracked)";
  if (heap->block_names)
  {
    uint64_t hash = hash_pool(pool);
    int32_t  pool_block_index = hashmap_get(heap->lookup, hash);
    if (pool_block_index != -1)
      name = heap->block_names[pool_block_index];
  }

  return name;
}

const char* xheap_block_name(const xheap_t heap, const void* pointer)
{
  const char* name = "(not tracked)";
  if (heap->block_names)
  {
    uint64_t hash        = hash_pointer(pointer);
    int32_t  block_index = hashmap_get(heap->lookup, hash);
    if (block_index != -1)
      name = heap->block_names[block_index];
  }

  return name;
}

int xheap_owns(const xheap_t heap, const void* pointer)
{
  return block_from_pointer(heap, pointer) ? 1 : 0;
}

int xheap_in_pool_range(const xheap_t heap, const xheap_pool_t pool, const void* pointer)
{
  pool_block_t* pool_block = pool_to_pool_block(heap, pool);
  if (!pool_block)
    return 0;

  uintptr_t ptr = (uintptr_t)pointer;
  uintptr_t bgn = (uintptr_t)pool_block->offset;
  uintptr_t end = (uintptr_t)pool_block->size + bgn;

  return ptr >= bgn && ptr < end;
}

int xheap_in_range(const xheap_t heap, const void* pointer)
{
  int32_t pool_block_index = heap->pools;
  while (pool_block_index != -1)
  {
    pool_block_t* pool_block = pool_block_from_index(heap, pool_block_index);

    uintptr_t ptr = (uintptr_t)pointer;
    uintptr_t bgn = (uintptr_t)pool_block->offset;
    uintptr_t end = (uintptr_t)pool_block->size + bgn;

    if (ptr >= bgn && ptr < end)
      return 1;

    pool_block_index = pool_block->pool_next;
  }

  return 0;
}

void* xheap_malloc(xheap_t heap, size_t bytes, const char* name)
{
	const size_t adjusted_size = tlsf_adjust_size(bytes, kMinAlignment);
  int32_t block_index = -1;
  block_t* block = freelist_alloc(heap, adjusted_size, &block_index);
  if (!block)
    return NULL;

	void* pointer = block_finalize_as_used(heap, block_index, block, adjusted_size, name);
  return pointer;
}

void* xheap_memalign(xheap_t heap, size_t align, size_t bytes, const char* name)
{
  const size_t adjusted_size         = tlsf_adjust_size(bytes, kMinAlignment);
  const size_t aligned_adjusted_size = adjusted_size + align;

  int32_t block_index = -1;
  block_t* block = freelist_alloc(heap, aligned_adjusted_size, &block_index);
  if (!block)
    return NULL;

  uintptr_t offset         = block->offset;
  uintptr_t aligned_offset = xAlignUp(offset, align);
  uintptr_t diff           = xAlignDown(aligned_offset - offset, kMinAlignment);

  /* Try to recover memory left unused by the alignment. */
  if (diff && diff > kMinBlockSize)
  {
    block_t* prev = block_split_to_prev(heap, block, block->size - diff);
    if (prev)
      freelist_insert(heap, prev);
  }

  void* pointer = block_finalize_as_used(heap, block_index, block, adjusted_size, name);
  return pointer;
}

void* xheap_realloc(xheap_t heap, void* pointer, size_t size, const char* name)
{
	void* final_pointer = NULL;

  if (pointer && size)
  {
		block_t* block = block_from_pointer(heap, pointer);
		block_t* next  = block_from_index(heap, block->contiguous_next);

		const size_t block_size      = block->size;
		const size_t merge_next_size = block_size + next->size;
		const size_t adjusted_size   = tlsf_adjust_size(size, kMinAlignment);

		/*  If the next contiguous block isn't fit for use, the current alloc 
        needs to be relocated.
    */
		if (adjusted_size > block_size && (!next->is_free || adjusted_size > merge_next_size))
		{
			final_pointer = xheap_malloc(heap, size, name);
			if (final_pointer)
			{
				memcpy(final_pointer, pointer, xMin(block_size, size));
				xheap_free(heap, pointer);
			}
		}
		else
		{
      /* First merge contiguously next block. */
			if (adjusted_size > block_size)
			{
        freelist_remove(heap, next);
				block_merge_next(heap, block);
			}

      /* Then return any excess. */
	    if (block_can_split(block, adjusted_size))
	    {
        block_t* new_excess = block_split_to_next(heap, block, adjusted_size);
        if (new_excess)
        {
          block_merge_next(heap, new_excess);
          freelist_insert(heap, new_excess);
        }
	    }

			final_pointer = block_finalize_as_used(heap, block_to_index(heap, block), block, adjusted_size, name);
		}
	}
	else if (!pointer)
		final_pointer = xheap_malloc(heap, size, name);
	else
		xheap_free(heap, pointer);

	return final_pointer;
}

void xheap_free(xheap_t heap, const void* pointer)
{
	if (!pointer)
    return;
	block_t* block = block_from_pointer(heap, pointer);
	xAssert(!block->is_free, "%p maps to block %p that is marked as already free (double-free?).", pointer, block);
  block->is_free = -1;
  block_merge_prev(heap, block);
  block_merge_next(heap, block);
  freelist_insert(heap, block);
}

int32_t xheap_validate(const xheap_t heap)
{
  int32_t nerrors = heap_validate_tlsf(heap);

  int32_t pool_block_index = heap->pools;
  while (pool_block_index != -1)
  {
    pool_block_t* pool_block = pool_block_from_index(heap, pool_block_index);
    if (pool_block)
      nerrors += pool_validate(heap, pool_block);
    else
      nerrors++;

    pool_block_index = pool_block->pool_next;
  }

  return nerrors;
}

typedef struct walk_ctx_t
{
  xheap_t heap;
  void* user;
} walk_ctx_t;

static void default_walker(walk_ctx_t* user, void* pointer, size_t size, int used)
{
  const block_t* block = block_from_pointer(user->heap, pointer);
  int32_t block_index = block_to_index(user->heap, block);

	xTrace("\t%p %s size: %x (%u, %p)\n", pointer, used ? "used" : "free", (uint32_t)size, block_index, block);
}

void xheap_walk_pool(const xheap_t heap, xheap_pool_t pool, void* user, xheap_walker walker)
{
  pool_block_t* pool_block = pool_to_pool_block(heap, pool);
  if (!pool_block)
    return;

  walk_ctx_t default_ctx;
  if (!walker)
  {
    walker = (xheap_walker)default_walker;
    default_ctx.heap = heap;
    default_ctx.user = user;
    user = &default_ctx;
  }

  int32_t       block_index = pool_block->first_block;
  const int32_t last_block  = pool_block->last_block;

  while (block_index != last_block)
  {
    block_t*    block = block_from_index(heap, block_index);
    const char* name  = block_name(heap, block_index);
    
    walker(user, (void*)block->offset, block->size, name, pool, !block->is_free);
    
    block_index = block->contiguous_next;
  }
}

void xheap_walk_pools(const xheap_t heap, void* user, xheap_walker walker)
{
  int32_t pool_block_index = heap->pools;
  while (pool_block_index != -1)
  {
    pool_block_t* pool_block = pool_block_from_index(heap, pool_block_index);
    if (pool_block)
    {
      const char* name = block_name          (heap, pool_block_index);
      int         used = pool_is_single_block(heap, pool_block);

      walker(user, (void*)pool_block->offset, pool_block->size, name, (xheap_pool_t)pool_block->offset, used);
    }

    pool_block_index = pool_block->pool_next;
  }
}

void xheap_walk(const xheap_t heap, void* user, xheap_walker walker)
{
  int32_t pool_block_index = heap->pools;
  while (pool_block_index != -1)
  {
    pool_block_t* pool_block = pool_block_from_index(heap, pool_block_index);
    const xheap_pool_t pool  = (const xheap_pool_t)pool_block->offset;

    xheap_walk_pool(heap, pool, user, walker);

    pool_block_index = pool_block->pool_next;
  }
}


/* Unit test */
#include <stdlib.h>

int xheap_unittest()
{
  static const size_t kMaxAllocs = 20;
  static const size_t kPoolSize  = 2 * 1024;

  if (firstbitlow(12) != 2)
    return 1;

  if (firstbithigh(12) != 3)
    return 2;

  if (firstbitlow(0) != -1)
    return 3;

  if (firstbithigh(0) != -1)
    return 4;

  if (firstbitlow(1) != 0)
    return 3;

  if (firstbithigh(1) != 0)
    return 4;

  #define kHighIndex (sizeof(size_t) * 8 - 1)
  #define kSomeLowerIndex 15
  static const size_t kHighBit        = ((size_t)1) << kHighIndex;
  static const size_t kSomeLowerBit   = ((size_t)1) << kSomeLowerIndex;

  if (firstbitlow(kHighBit|kSomeLowerBit) != kSomeLowerIndex)
    return 5;

  if (firstbithigh(kHighBit|kSomeLowerBit) != kHighIndex)
    return 6;

  if (log2u(0) != 0)
    return 7;

  if (log2u(8) != 3)
    return 8;
  
  const int kUseNaming = 1;
  const size_t bookkeeping_bytes = xheap_required_bytes(kMaxAllocs, kUseNaming);

  size_t max_allocs = xheap_calc_max_allocation_count(bookkeeping_bytes, kUseNaming);
  if (max_allocs != kMaxAllocs)
    return -1;

  void* bookkeeping_mem = malloc(bookkeeping_bytes);
  void* pool_mem        = malloc(kPoolSize);

  xheap_t heap = xheap_init(bookkeeping_mem, bookkeeping_bytes, kUseNaming);
  if (!heap)
    return -2;

  xheap_pool_t pool = xheap_add_pool(heap, pool_mem, kPoolSize, "testpool");
  if (!pool)
    return -3;

  const char* testname = xheap_pool_name(heap, pool);
  if (strcmp(testname, "testpool"))
    return -4;

  void* a = xheap_malloc(heap, 1024, "1024");
  if (!a)
    return -5;

  void* b = xheap_malloc(heap, 1024, NULL);
  if (!b)
    return -6;

  if (((char*)b - (char*)a) != 1024)
    return -7;

  void* should_be_null = xheap_malloc(heap, 1, "Fail");
  if (should_be_null != 0)
    return -8;

  xheap_free(heap, a);

  a = xheap_malloc(heap, 512, "test named alloc");
  if (!a)
    return -9;

  void* c = xheap_malloc(heap, 512, NULL);
  if (!c)
    return -10;

  size_t sz = xheap_block_size(heap, c);
  if (sz != 512)
    return -11;

  testname = xheap_block_name(heap, a);
  if (strcmp(testname, "test named alloc"))
    return -12;

  xheap_free(heap, b);
  xheap_free(heap, c);
  xheap_free(heap, a);

  a = xheap_malloc(heap, 2048, NULL);
  if (!a)
    return -13;

  xheap_realloc(heap, a, 0, NULL);

  a = xheap_memalign(heap, 64, 13, NULL);
  if (!xAligned(a, 64))
    return -14;

  b = xheap_memalign(heap, 128, 257, NULL);
  if (!xAligned(a, 128))
    return -15;

  c = xheap_memalign(heap, 512, 257, NULL);
  if (!xAligned(a, 512))
    return -16;

  xheap_free(heap, c);
  xheap_free(heap, a);
  xheap_free(heap, b);

  xheap_remove_pool(heap, pool);

  free(pool_mem);
  free(bookkeeping_mem);
  return 0;
}
