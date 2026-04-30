from collections import defaultdict
from pathlib import Path


DATA_DIR = Path("data/MetaQA_fixed")  # change if needed

TRAIN = DATA_DIR / "train.txt"
VALID = DATA_DIR / "valid.txt"
TEST = DATA_DIR / "test.txt"


def load_triples(path: Path):
    triples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no} expected 3 columns, got {len(parts)}: {line!r}")
            h, r, t = parts
            triples.append((h, r, t))
    return triples


def build_hr_to_tails(triples):
    hr_to_tails = defaultdict(set)
    for h, r, t in triples:
        hr_to_tails[(h, r)].add(t)
    return hr_to_tails


def report_conflicts(train, valid, test):
    train_hr = build_hr_to_tails(train)
    valid_hr = build_hr_to_tails(valid)
    test_hr = build_hr_to_tails(test)

    def compare(name, left_hr, right_hr):
        shared_hr = set(left_hr) & set(right_hr)

        conflict_count = 0
        examples = []

        for hr in shared_hr:
            left_tails = left_hr[hr]
            right_tails = right_hr[hr]

            # same (h,r) appears in both, but tails differ
            if left_tails != right_tails:
                conflict_count += 1
                if len(examples) < 10:
                    examples.append((hr, sorted(list(left_tails))[:5], sorted(list(right_tails))[:5]))

        print(f"\n{name}")
        print(f"shared (h,r) pairs: {len(shared_hr)}")
        print(f"conflicting (h,r) pairs: {conflict_count}")

        if examples:
            print("examples:")
            for (h, r), left_tails, right_tails in examples:
                print(f"  ({h}, {r})")
                print(f"    left tails : {left_tails}")
                print(f"    right tails: {right_tails}")

    print("Split sizes:")
    print(f"  train triples: {len(train)}")
    print(f"  valid triples: {len(valid)}")
    print(f"  test triples : {len(test)}")

    compare("TRAIN vs VALID", train_hr, valid_hr)
    compare("TRAIN vs TEST", train_hr, test_hr)
    compare("VALID vs TEST", valid_hr, test_hr)


def main():
    train = load_triples(TRAIN)
    valid = load_triples(VALID)
    test = load_triples(TEST)

    report_conflicts(train, valid, test)


if __name__ == "__main__":
    main()