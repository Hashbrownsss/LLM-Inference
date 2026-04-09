"""
KV Cache Manager — manages GPU memory for key-value cache across sequences.

This is the core innovation that made vLLM famous (PagedAttention).

Problem it solves:
- Naive approach: allocate one big contiguous block for each sequence's KV cache
- Problem: sequences have different lengths, causing memory fragmentation
- After a sequence finishes, its memory block sits mostly unused (waste)
- Result: can only fit fewer sequences in GPU memory than hardware allows

Solution (PagedAttention-style):
- Allocate KV cache in fixed-size blocks (e.g., 16 tokens per block)
- Use a page table to map logical token positions to physical blocks
- Like OS virtual memory — contiguous virtual addresses, scattered physical pages
- Memory utilization: ~60-80% wasted -> <4% wasted

This implementation is a simplified educational version.
Production: see vLLM's paged_attention_cache.py
"""

import torch
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Configuration
# =============================================================================

# Number of tokens per KV cache block
# Smaller = less waste, more page table lookups
# Larger = less overhead, more waste for short sequences
BLOCK_SIZE = 16

# Number of blocks to pre-allocate per sequence
INITIAL_BLOCKS_PER_SEQUENCE = 4


# =============================================================================
# Page Table Entry
# =============================================================================

@dataclass
class Block:
    """
    A single block of KV cache memory.

    Each block holds BLOCK_SIZE tokens' worth of K and V tensors.
    When a block fills up, we allocate a new one.
    """
    physical_block_id: int       # Where this block lives in GPU memory
    token_start: int             # First logical token position this block covers
    token_end: int               # Last logical token position (exclusive)
    is_full: bool = False
    num_tokens: int = 0          # How many tokens currently stored in this block

    @property
    def free_slots(self) -> int:
        return BLOCK_SIZE - self.num_tokens


# =============================================================================
# Page Table
# =============================================================================

class KVCachePageTable:
    """
    Maps logical token positions to physical KV cache blocks.

    Think of it like a CPU's page table:
    - Logical view: sequence has tokens at positions 0, 1, 2, ... N
    - Physical view: blocks scattered across GPU memory at various addresses

    page_table[logical_position] = physical_block_id

    Example:
        Sequence "What is AI?" (5 tokens)
        Logical positions:  [0, 1, 2, 3, 4]
        Block 0 (ids 0-15):  [0, 1, 2, 3, 4]  <- all 5 tokens fit in one block
        page_table = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        Sequence grows to 20 tokens:
        Block 0 (ids 0-15):  [0-15]
        Block 1 (ids 16-31): [16-19]  <- partially filled
        page_table = {0:0, 1:0, ..., 15:0, 16:1, 17:1, 18:1, 19:1}
    """

    def __init__(self, max_blocks: int, device: str = "cuda"):
        """
        Args:
            max_blocks: Total number of blocks we can track (GPU memory budget)
            device: Where KV cache tensors live
        """
        self.max_blocks = max_blocks
        self.device = device

        # Allocate physical block storage
        # Shape: [max_blocks, num_layers, num_heads, block_size, head_dim]
        # This is a simplification — real implementation uses dynamic allocation
        self._num_layers = 22     # TinyLlama has 22 layers
        self._num_heads = 16     # TinyLlama has 16 heads
        self._head_dim = 128    # Each head is 128 dimensions

        try:
            self.k_blocks = torch.zeros(
                max_blocks, self._num_layers, self._num_heads, BLOCK_SIZE, self._head_dim,
                dtype=torch.float16, device=device
            )
            self.v_blocks = torch.zeros_like(self.k_blocks)
        except (RuntimeError, AssertionError):
            # Not enough GPU memory — fall back gracefully
            self.k_blocks = None
            self.v_blocks = None

        # Free block pool — block IDs available for allocation
        self._free_blocks: set[int] = set(range(max_blocks))
        # Allocated blocks per sequence
        self._sequences: dict[int, list[Block]] = {}  # sequence_id -> list of blocks
        # Sequence counter
        self._next_seq_id = 0

    def allocate_sequence(self) -> int:
        """
        Allocate a new sequence in the KV cache.
        Returns a sequence ID.
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        # Allocate initial blocks from free pool
        blocks = []
        for _ in range(INITIAL_BLOCKS_PER_SEQUENCE):
            if not self._free_blocks:
                # Out of blocks — evict the oldest sequence
                self._evict_oldest_sequence()
            block_id = self._free_blocks.pop()
            blocks.append(Block(
                physical_block_id=block_id,
                token_start=len(blocks) * BLOCK_SIZE,
                token_end=(len(blocks) + 1) * BLOCK_SIZE,
            ))

        self._sequences[seq_id] = blocks
        return seq_id

    def free_sequence(self, seq_id: int):
        """
        Free all blocks belonging to a sequence.
        Called when a sequence finishes generating.
        """
        if seq_id not in self._sequences:
            return

        for block in self._sequences[seq_id]:
            self._free_blocks.add(block.physical_block_id)

        del self._sequences[seq_id]

    def append_tokens(self, seq_id: int, num_new_tokens: int) -> bool:
        """
        Extend a sequence by num_new_tokens.
        Allocates new blocks if needed.
        Returns True if successful, False if out of memory.
        """
        if seq_id not in self._sequences:
            return False

        blocks = self._sequences[seq_id]
        current_block = blocks[-1]

        if current_block.free_slots >= num_new_tokens:
            # Fits in current block
            current_block.num_tokens += num_new_tokens
            return True

        # Need new blocks
        tokens_to_allocate = num_new_tokens - current_block.free_slots
        current_block.num_tokens = BLOCK_SIZE
        current_block.is_full = True

        while tokens_to_allocate > 0:
            if not self._free_blocks:
                self._evict_oldest_sequence()
                if not self._free_blocks:
                    return False  # Truly out of memory

            block_id = self._free_blocks.pop()
            num_in_new_block = min(BLOCK_SIZE, tokens_to_allocate)
            new_block = Block(
                physical_block_id=block_id,
                token_start=blocks[-1].token_end,
                token_end=blocks[-1].token_end + BLOCK_SIZE,
                num_tokens=num_in_new_block,
                is_full=num_in_new_block == BLOCK_SIZE,
            )
            blocks.append(new_block)
            tokens_to_allocate -= num_in_new_block

        return True

    def get_block_for_position(self, seq_id: int, position: int) -> Optional[Block]:
        """Get the physical block that holds token at logical position."""
        if seq_id not in self._sequences:
            return None

        for block in self._sequences[seq_id]:
            if block.token_start <= position < block.token_end:
                return block
        return None

    def get_allocated_blocks(self) -> int:
        """Number of blocks currently in use."""
        return self._max_blocks_ever_used() - len(self._free_blocks)

    def get_memory_usage_mb(self) -> float:
        """Estimated GPU memory used by KV cache in MB."""
        if self.k_blocks is None:
            return 0.0
        bytes_per_block = (
            2 *  # K and V
            self._num_layers *
            self._num_heads *
            BLOCK_SIZE *
            self._head_dim *
            2  # float16 = 2 bytes
        )
        allocated = self.get_allocated_blocks()
        return (bytes_per_block * allocated) / (1024 * 1024)

    def get_utilization(self) -> float:
        """What fraction of allocated blocks are actually used?"""
        if self.get_allocated_blocks() == 0:
            return 0.0

        total_slots = self.get_allocated_blocks() * BLOCK_SIZE
        used_slots = 0
        for blocks in self._sequences.values():
            used_slots += sum(b.num_tokens for b in blocks)
        return used_slots / total_slots if total_slots > 0 else 0.0

    def _evict_oldest_sequence(self):
        """Evict the first (oldest) sequence to free blocks."""
        if not self._sequences:
            return
        oldest_seq_id = min(self._sequences.keys())
        self.free_sequence(oldest_seq_id)

    def _max_blocks_ever_used(self) -> int:
        return self.max_blocks

    def get_status(self) -> dict:
        """Return diagnostic information about the cache."""
        return {
            "max_blocks": self.max_blocks,
            "allocated_blocks": self.get_allocated_blocks(),
            "free_blocks": len(self._free_blocks),
            "active_sequences": len(self._sequences),
            "memory_used_mb": round(self.get_memory_usage_mb(), 1),
            "utilization": round(self.get_utilization() * 100, 1),
            "block_size": BLOCK_SIZE,
        }
