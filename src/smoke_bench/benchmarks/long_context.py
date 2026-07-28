"""Needle-in-a-haystack: hide a fact in a long passage and ask the model to
retrieve it. Tests effective context window."""

from __future__ import annotations

import random
from dataclasses import dataclass

from smoke_bench.benchmarks.base import Benchmark, Sample

NEEDLE = "The secret code word for this test is BLUEBIRD-7421."
QUESTION = "What is the secret code word mentioned in the passage?"

# Filler paragraphs used to pad context to the desired size.
FILLER_PARAGRAPHS = [
    "The old lighthouse stood at the edge of the cliff, its beam sweeping the dark water every night. "
    "Sailors relied on it for generations, until modern GPS made its service less critical. "
    "Still, the keeper kept the light burning, a quiet ritual that anchored the community.",
    "Beneath the city, a network of tunnels wound through ancient stone. Most had been sealed "
    "centuries ago, but a few remained accessible to those who knew the patterns. Scholars debated "
    "their original purpose: escape routes, smuggling paths, or simple infrastructure.",
    "The train slowed as it entered the mountain pass. Outside, snow fell in slow, heavy flakes, "
    "obscuring the valley below. The conductor announced a brief delay and passengers settled in, "
    "some reading, others staring out at the white world beyond the glass.",
    "The market square bustled on Saturday mornings. Farmers brought crates of apples and greens, "
    "fishmongers arranged their ice, and bakers sold sourdough still warm from the oven. Children "
    "weaved between stalls with coins clutched in small fists.",
    "In the observatory, astronomers tracked an asteroid that would pass close to Earth in 2061. "
    "The trajectory was well understood, but the team still met weekly to refine the calculations. "
    "Public outreach was part of the mission: people were less afraid when they understood the math.",
    "The painter mixed ultramarine with white, testing the gradient on a scrap of canvas. The blue "
    "needed to evoke the evening sky just after sunset, when color drains from the horizon but the "
    "memory of day lingers in the west. She worked in silence, the studio dim save for a single lamp.",
    "Engineers gathered around the prototype, taking notes as the device whirred. The new motor was "
    "smaller and more efficient than its predecessor, but there was a worrying vibration at high RPM. "
    "One of them suggested a counterweight; another proposed a different bearing material. They "
    "agreed to test both.",
    "The library smelled of paper and dust. Rows of volumes stretched toward the ceiling, each spine "
    "a small monument to some long-ago idea. A scholar traced her finger along the shelves until she "
    "found the volume she wanted, then settled at a reading desk and began to turn the pages.",
]

_INSTRUCTION = (
    "Read the passage below carefully. When asked a question, answer using only "
    "information from the passage. If the answer is not present, say so.\n\n"
)


@dataclass
class _Config:
    target_words: int
    label: str


CONFIGS: list[_Config] = [
    _Config(target_words=800, label="~1k"),
    _Config(target_words=6500, label="~8k"),
    _Config(target_words=26000, label="~32k"),
    _Config(target_words=52000, label="~64k"),
    _Config(target_words=105000, label="~128k"),
]


def _make_passage(target_words: int, seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    out: list[str] = []
    word_count = 0
    # Randomly insert the needle in the middle ~50% of the passage.
    insert_at = rng.randint(max(1, target_words // 3), max(2, 2 * target_words // 3))
    while word_count < target_words:
        para = rng.choice(FILLER_PARAGRAPHS)
        out.append(para)
        word_count += len(para.split())
        # Plant the needle at the chosen position.
        if insert_at and word_count >= insert_at and "BLUEBIRD" not in " ".join(out):
            out.append(NEEDLE)
            word_count += len(NEEDLE.split())
            insert_at = 0
    return " ".join(out), word_count


class LongContextBenchmark(Benchmark):
    name = "long_context"
    description = "Needle-in-a-haystack: extract a planted fact at varying context sizes."

    def __init__(self, n_samples: int | None = None) -> None:
        super().__init__(n_samples)
        self._samples: list[Sample] | None = None

    @property
    def samples(self) -> list[Sample]:
        if self._samples is not None:
            return self._samples
        out: list[Sample] = []
        for idx, cfg in enumerate(CONFIGS):
            passage, n_words = _make_passage(cfg.target_words, seed=42 + idx)
            prompt = _INSTRUCTION + passage + "\n\nQuestion: " + QUESTION
            out.append(
                Sample(
                    id=f"needle_{cfg.label}",
                    prompt=prompt,
                    expected="BLUEBIRD-7421",
                    grader="contains",
                    request_kwargs={"max_tokens": 512, "temperature": 0.0},
                    tags={"context_words": n_words, "context_label": cfg.label},
                )
            )
        self._samples = out
        return out
