"""
core/frame_deduplicator.py

Duplicate frame detection for the vivarium ingestion pipeline.

How it works
────────────
Each incoming frame is fingerprinted with two hashes:

  1. MD5 of raw bytes — catches exact byte-for-byte duplicates (same file
     re-uploaded, S3 retransmit, client retry).

  2. Perceptual hash (pHash, 8x8 DCT) — catches visually identical frames
     that differ only in JPEG re-compression, metadata, or minor encoding
     artefacts. Two frames are considered perceptually duplicate when their
     Hamming distance is <= phash_distance_threshold (default 4/64 bits).

Scope is per cage_id: the same image sent for two different cages is NOT
a duplicate. This matches the real use-case where multiple cameras capture
different cages simultaneously.

A fixed-size LRU cache per cage keeps the last `cache_size` hashes in
memory. Frames older than `ttl_seconds` are expired on each insert so the
cache doesn't grow unbounded during long runs.

Factory pattern
───────────────
Use DeduplicatorFactory.create(config) — never instantiate directly.
Currently only one strategy ("phash") is supported; the factory allows
new strategies (e.g. "ssim", "none") to be added without touching the
orchestrator.

Config keys (under deduplication: in config.yaml):
  enabled:                  true     master switch (default: false)
  strategy:                 phash    which deduplicator to use (default: phash)
  cache_size:               64       max recent hashes per cage
  ttl_seconds:              300      max age of a cached hash (5 min)
  phash_distance_threshold: 4        max Hamming distance to flag duplicate

Usage:
  from core.frame_deduplicator import DeduplicatorFactory
  dedup = DeduplicatorFactory.create(config)          # once at init
  is_dup, reason = dedup.check(cage_id, image_bytes)
"""

from __future__ import annotations

import hashlib
import importlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Tuple

import cv2
import numpy as np
from dotmap import DotMap
from loguru import logger


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseDeduplicator(ABC):
    """Abstract duplicate-frame detector. All strategies must subclass this."""

    @abstractmethod
    def check(self, cage_id: str, image_bytes: bytes) -> Tuple[bool, str]:
        """
        Returns (is_duplicate, reason_string).
        reason is empty string when not a duplicate.
        """
        ...

    def stats(self) -> Dict[str, int]:
        """Return per-cage cache sizes. Override in subclasses that cache."""
        return {}


# ── No-op deduplicator (strategy: "none") ────────────────────────────────────

class NoopDeduplicator(BaseDeduplicator):
    """Deduplication disabled — every frame passes through."""

    def check(self, cage_id: str, image_bytes: bytes) -> Tuple[bool, str]:
        return False, ""


# ── pHash deduplicator (strategy: "phash") ───────────────────────────────────

def _phash(image_bytes: bytes, hash_size: int = 8) -> int:
    """
    Compute a perceptual hash (pHash) of an image from raw bytes.

    Steps:
      1. Decode image (any format OpenCV supports)
      2. Resize to (hash_size*4) x (hash_size*4) in grayscale
      3. Apply 2-D DCT and take top-left hash_size x hash_size coefficients
      4. Threshold at median → 64-bit integer bitmask

    Returns a 64-bit integer. Returns 0 on decode failure (treated as
    non-duplicate so a corrupt frame is still passed to the validator).
    """
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    size = hash_size * 4
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    flt = small.astype(np.float32)

    dct = cv2.dct(flt)
    top_left = dct[:hash_size, :hash_size]

    median = float(np.median(top_left))
    bits = (top_left > median).flatten()

    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class _Entry:
    md5: str
    phash: int
    inserted_at: float = field(default_factory=time.monotonic)


class PHashDeduplicator(BaseDeduplicator):
    """
    MD5 + perceptual-hash duplicate detector with per-cage LRU cache.

    Registered in DeduplicatorFactory as strategy: "phash".
    """

    def __init__(self, config: DotMap) -> None:
        dedup_cfg = getattr(config, "deduplication", DotMap())
        self._cache_size: int = int(getattr(dedup_cfg, "cache_size", 64))
        self._ttl: float = float(getattr(dedup_cfg, "ttl_seconds", 300))
        self._phash_threshold: int = int(getattr(dedup_cfg, "phash_distance_threshold", 4))
        self._caches: Dict[str, OrderedDict] = {}

        logger.info(
            f"PHashDeduplicator ready | "
            f"cache_size={self._cache_size} ttl={self._ttl}s "
            f"phash_threshold={self._phash_threshold}"
        )

    def check(self, cage_id: str, image_bytes: bytes) -> Tuple[bool, str]:
        md5 = hashlib.md5(image_bytes).hexdigest()
        ph = _phash(image_bytes)

        cache = self._get_cache(cage_id)
        self._expire(cache)

        # Check 1 — exact byte duplicate
        if md5 in cache:
            reason = f"exact duplicate (md5={md5[:8]}…)"
            logger.info(f"[Dedup] cage={cage_id} | {reason}")
            return True, reason

        # Check 2 — perceptual duplicate
        if ph != 0:
            for entry in cache.values():
                dist = _hamming(ph, entry.phash)
                if dist <= self._phash_threshold:
                    reason = (
                        f"perceptual duplicate (hamming={dist} "
                        f"<= threshold={self._phash_threshold})"
                    )
                    logger.info(f"[Dedup] cage={cage_id} | {reason}")
                    return True, reason

        # Not a duplicate — register
        self._insert(cache, md5, _Entry(md5=md5, phash=ph))
        return False, ""

    def stats(self) -> Dict[str, int]:
        return {cage_id: len(c) for cage_id, c in self._caches.items()}

    def _get_cache(self, cage_id: str) -> OrderedDict:
        if cage_id not in self._caches:
            self._caches[cage_id] = OrderedDict()
        return self._caches[cage_id]

    def _expire(self, cache: OrderedDict) -> None:
        now = time.monotonic()
        expired = [k for k, v in cache.items() if (now - v.inserted_at) > self._ttl]
        for k in expired:
            del cache[k]

    def _insert(self, cache: OrderedDict, key: str, entry: _Entry) -> None:
        if len(cache) >= self._cache_size:
            cache.popitem(last=False)
        cache[key] = entry


# ── Factory ───────────────────────────────────────────────────────────────────

class DeduplicatorFactory:
    """
    Creates a BaseDeduplicator from config.deduplication.strategy.

    Registry maps strategy name → (module_path, class_name).
    Add new strategies here without touching the orchestrator.
    """

    _REGISTRY: Dict[str, Tuple[str, str]] = {
        "phash": (__name__, "PHashDeduplicator"),
        "none":  (__name__, "NoopDeduplicator"),
    }

    @classmethod
    def create(cls, config: DotMap) -> BaseDeduplicator:
        dedup_cfg = getattr(config, "deduplication", DotMap())
        enabled: bool = bool(getattr(dedup_cfg, "enabled", False))

        if not enabled:
            logger.info("DeduplicatorFactory | deduplication disabled → NoopDeduplicator")
            return NoopDeduplicator()

        strategy: str = str(getattr(dedup_cfg, "strategy", "phash")).lower().strip()

        if strategy not in cls._REGISTRY:
            known = ", ".join(sorted(cls._REGISTRY.keys()))
            raise ValueError(
                f"Unknown deduplication strategy '{strategy}'. "
                f"Known strategies: {known}. "
                f"Check config.yaml → deduplication.strategy"
            )

        module_path, class_name = cls._REGISTRY[strategy]
        module = importlib.import_module(module_path)
        klass = getattr(module, class_name)

        logger.info(f"DeduplicatorFactory | strategy={strategy} → {class_name}")
        return klass(config)

    @classmethod
    def register(cls, key: str, module_path: str, class_name: str) -> None:
        """Register a custom deduplication strategy at runtime."""
        cls._REGISTRY[key] = (module_path, class_name)
